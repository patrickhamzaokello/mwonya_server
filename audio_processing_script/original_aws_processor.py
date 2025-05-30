import os
import json
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime

# S3 and AWS imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

class S3AudioProcessor:
    """Processes audio files from S3 for adaptive streaming"""

    # Quality configurations optimized for music streaming
    QUALITY_CONFIGS = {
        'low': {
            'bitrate': '64k',
            'sample_rate': '22050',
            'channels': '2',
            'description': 'Low quality for slow connections'
        },
        'med': {
            'bitrate': '128k',
            'sample_rate': '44100',
            'channels': '2',
            'description': 'Standard quality'
        },
        'high': {
            'bitrate': '320k',
            'sample_rate': '44100',
            'channels': '2',
            'description': 'High quality'
        }
    }

    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}

    def __init__(self,
                 source_bucket: str,
                 dest_bucket: str,
                 source_prefix: str = '',
                 dest_prefix: str = 'processed_audio',
                 segment_duration: int = 10,
                 region_name: str = 'us-east-1',
                 max_workers: int = 2):
        """
        Initialize the S3 audio processor

        Args:
            source_bucket: S3 bucket containing source audio files
            dest_bucket: S3 bucket for processed files (can be same as source)
            source_prefix: Prefix/folder in source bucket to process
            dest_prefix: Prefix/folder in destination bucket for results
            segment_duration: Duration of each HLS segment in seconds
            region_name: AWS region
            max_workers: Maximum concurrent workers (keep low for large batches)
        """
        self.source_bucket = source_bucket
        self.dest_bucket = dest_bucket
        self.source_prefix = source_prefix.strip('/')
        self.dest_prefix = dest_prefix.strip('/')
        self.segment_duration = segment_duration
        self.max_workers = max_workers

        # Configure S3 client with retry logic
        config = Config(
            region_name=region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )

        try:
            self.s3_client = boto3.client('s3', config=config)
            # Test S3 connection
            self.s3_client.head_bucket(Bucket=source_bucket)
            print(f"✅ Connected to S3 bucket: {source_bucket}")
        except NoCredentialsError:
            raise RuntimeError("AWS credentials not configured. Use 'aws configure' or set environment variables.")
        except ClientError as e:
            raise RuntimeError(f"Failed to connect to S3 bucket {source_bucket}: {e}")

        # Check dependencies
        self._check_dependencies()

        # Create temporary directory for processing
        self.temp_dir = Path(tempfile.mkdtemp(prefix='s3_audio_processor_'))
        print(f"📁 Using temporary directory: {self.temp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory")

    def _check_dependencies(self) -> None:
        """Check if ffmpeg and ffprobe are available"""
        for tool in ['ffmpeg', 'ffprobe']:
            try:
                subprocess.run([tool, '-version'],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(f"{tool} is not installed or not in PATH")

        print("✅ Dependencies check passed")

    def list_audio_files(self) -> List[str]:
        """
        List all audio files in the S3 bucket with the specified prefix

        Returns:
            List of S3 keys for audio files
        """
        audio_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            page_iterator = paginator.paginate(
                Bucket=self.source_bucket,
                Prefix=self.source_prefix
            )

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Check if it's an audio file
                        if any(key.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
                            audio_files.append(key)

            print(f"📁 Found {len(audio_files)} audio files in s3://{self.source_bucket}/{self.source_prefix}")
            return audio_files

        except ClientError as e:
            print(f"❌ Error listing S3 objects: {e}")
            return []

    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download a file from S3 to local path

        Args:
            s3_key: S3 object key
            local_path: Local file path to download to

        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.source_bucket, s3_key, str(local_path))
            return True
        except ClientError as e:
            print(f"❌ Failed to download {s3_key}: {e}")
            return False

    def upload_directory(self, local_dir: Path, s3_prefix: str) -> bool:
        """
        Upload a directory and all its contents to S3

        Args:
            local_dir: Local directory to upload
            s3_prefix: S3 prefix/folder to upload to

        Returns:
            True if successful, False otherwise
        """
        try:
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path from local_dir
                    relative_path = file_path.relative_to(local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')

                    # Upload file
                    self.s3_client.upload_file(
                        str(file_path),
                        self.dest_bucket,
                        s3_key,
                        ExtraArgs={'ContentType': self._get_content_type(file_path)}
                    )

            return True

        except ClientError as e:
            print(f"❌ Failed to upload directory {local_dir}: {e}")
            return False

    def _get_content_type(self, file_path: Path) -> str:
        """Get appropriate content type for file"""
        suffix = file_path.suffix.lower()
        content_types = {
            '.m3u8': 'application/vnd.apple.mpegurl',
            '.ts': 'video/mp2t',
            '.json': 'application/json',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg'
        }
        return content_types.get(suffix, 'application/octet-stream')

    def check_if_processed(self, track_id: str) -> bool:
        """
        Check if a track has already been processed in S3

        Args:
            track_id: Track identifier

        Returns:
            True if already processed, False otherwise
        """
        try:
            # Check for master playlist
            master_key = f"{self.dest_prefix}/{track_id}/playlist.m3u8"
            self.s3_client.head_object(Bucket=self.dest_bucket, Key=master_key)

            # Check for metadata
            metadata_key = f"{self.dest_prefix}/{track_id}/metadata.json"
            self.s3_client.head_object(Bucket=self.dest_bucket, Key=metadata_key)

            # Check for quality playlists
            for quality in self.QUALITY_CONFIGS.keys():
                quality_key = f"{self.dest_prefix}/{track_id}/{quality}/playlist.m3u8"
                self.s3_client.head_object(Bucket=self.dest_bucket, Key=quality_key)

            return True

        except ClientError:
            return False

    def get_audio_info(self, input_path: Path) -> Dict:
        """
        Get audio file information using ffprobe

        Args:
            input_path: Path to the audio file

        Returns:
            Dictionary containing audio metadata
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(input_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Extract relevant information
            format_info = data.get('format', {})
            audio_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
                {}
            )

            return {
                'duration': float(format_info.get('duration', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'title': format_info.get('tags', {}).get('title', input_path.stem),
                'artist': format_info.get('tags', {}).get('artist', 'Unknown Artist'),
                'album': format_info.get('tags', {}).get('album', ''),
                'sample_rate': int(audio_stream.get('sample_rate', 44100)),
                'channels': int(audio_stream.get('channels', 2)),
                'codec': audio_stream.get('codec_name', 'unknown')
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            print(f"Failed to get audio info for {input_path}: {e}")
            return {
                'duration': 180.0,  # Default 3 minutes
                'bitrate': 128000,
                'title': input_path.stem,
                'artist': 'Unknown Artist',
                'album': '',
                'sample_rate': 44100,
                'channels': 2,
                'codec': 'unknown'
            }

    def generate_track_id(self, s3_key: str, metadata: Dict) -> str:
        """
        Generate a unique track ID based on S3 key and metadata

        Args:
            s3_key: S3 object key
            metadata: Audio metadata

        Returns:
            Unique track ID
        """
        hasher = hashlib.md5()
        hasher.update(s3_key.encode())
        hasher.update(metadata['title'].encode())
        hasher.update(metadata['artist'].encode())

        return f"track_{hasher.hexdigest()[:12]}"

    def process_quality(self, input_path: Path, output_dir: Path,
                       quality: str, config: Dict) -> bool:
        """
        Process audio file for a specific quality level

        Args:
            input_path: Path to input audio file
            output_dir: Output directory for this quality
            quality: Quality level name
            config: Quality configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            playlist_path = output_dir / 'playlist.m3u8'

            # FFmpeg command optimized for audio streaming
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-c:a', 'aac',                              # Use AAC codec for better compatibility
                '-b:a', config['bitrate'],                  # Audio bitrate
                '-ar', config['sample_rate'],               # Sample rate
                '-ac', config['channels'],                  # Audio channels
                '-profile:a', 'aac_low',                    # AAC profile for better compatibility
                '-movflags', '+faststart',                  # Optimize for streaming
                '-f', 'hls',                                # HLS format
                '-hls_time', str(self.segment_duration),    # Segment duration
                '-hls_list_size', '0',                      # Keep all segments in playlist
                '-hls_segment_filename', str(output_dir / 'segment_%03d.ts'),
                '-hls_playlist_type', 'vod',                # Video on demand
                '-hls_flags', 'independent_segments',       # Each segment is independent
                '-y',                                       # Overwrite output files
                str(playlist_path)
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Verify playlist was created
            if not playlist_path.exists():
                raise RuntimeError("Playlist file was not created")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed for {quality} quality: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Error processing {quality} quality: {e}")
            return False

    def generate_master_playlist(self, track_id: str) -> str:
        """
        Generate master playlist content for adaptive streaming

        Args:
            track_id: Track identifier

        Returns:
            Master playlist content
        """
        base_path = f"/{self.dest_prefix}/{track_id}"

        playlist_content = "#EXTM3U\n#EXT-X-VERSION:3\n\n"

        # Add each quality level with appropriate bandwidth values
        bandwidth_map = {
            'low': 80000,    # 64k audio + overhead
            'med': 160000,   # 128k audio + overhead
            'high': 384000   # 320k audio + overhead
        }

        for quality in ['low', 'med', 'high']:
            config = self.QUALITY_CONFIGS[quality]
            bandwidth = bandwidth_map[quality]

            playlist_content += f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS="mp4a.40.2"\n'
            playlist_content += f'{base_path}/{quality}/playlist.m3u8\n\n'

        return playlist_content

    def process_single_file(self, s3_key: str, force: bool = False) -> Optional[str]:
        """
        Process a single audio file from S3

        Args:
            s3_key: S3 key of the audio file
            force: If True, reprocess even if already exists

        Returns:
            Track ID if successful, None otherwise
        """
        file_name = Path(s3_key).name
        print(f"🎵 Processing: {file_name}")

        # Create temporary file for download
        temp_input = self.temp_dir / f"input_{hash(s3_key) % 10000}.tmp"
        temp_output = self.temp_dir / f"output_{hash(s3_key) % 10000}"

        try:
            # Download file from S3
            if not self.download_file(s3_key, temp_input):
                return None

            # Get audio information
            metadata = self.get_audio_info(temp_input)
            track_id = self.generate_track_id(s3_key, metadata)

            # Check if already processed (unless forced)
            if not force and self.check_if_processed(track_id):
                print(f"⏭️  Already processed: {file_name} ({track_id})")
                return track_id

            # Create local output directory
            temp_output.mkdir(parents=True, exist_ok=True)

            # Process each quality level
            success_count = 0
            for quality, config in self.QUALITY_CONFIGS.items():
                quality_dir = temp_output / quality
                if self.process_quality(temp_input, quality_dir, quality, config):
                    success_count += 1

            if success_count == 0:
                print(f"❌ Failed to process any quality for {file_name}")
                return None

            # Generate master playlist
            master_playlist_path = temp_output / 'playlist.m3u8'
            master_content = self.generate_master_playlist(track_id)

            with open(master_playlist_path, 'w') as f:
                f.write(master_content)

            # Save metadata
            metadata_file = temp_output / 'metadata.json'
            track_metadata = {
                'id': track_id,
                'title': metadata['title'],
                'artist': metadata['artist'],
                'album': metadata['album'],
                'duration': int(metadata['duration']),
                'original_s3_key': s3_key,
                'processed_at': datetime.now().isoformat(),
                'qualities': list(self.QUALITY_CONFIGS.keys()),
                'original_info': {
                    'bitrate': metadata['bitrate'],
                    'sample_rate': metadata['sample_rate'],
                    'channels': metadata['channels'],
                    'codec': metadata['codec']
                }
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(track_metadata, f, indent=2, ensure_ascii=False)

            # Upload processed files to S3
            s3_prefix = f"{self.dest_prefix}/{track_id}"
            if self.upload_directory(temp_output, s3_prefix):
                print(f"✅ Successfully processed: {file_name} ({track_id})")
                return track_id
            else:
                print(f"❌ Failed to upload processed files for {file_name}")
                return None

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            return None
        finally:
            # Cleanup temporary files
            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                shutil.rmtree(temp_output)

    def process_batch(self, s3_keys: List[str], force: bool = False) -> List[str]:
        """
        Process a batch of audio files from S3

        Args:
            s3_keys: List of S3 keys to process
            force: If True, reprocess even if files already exist

        Returns:
            List of successfully processed track IDs
        """
        if not s3_keys:
            print("No files to process")
            return []

        print(f"📁 Processing {len(s3_keys)} audio files with {self.max_workers} workers")

        processed_tracks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self.process_single_file, s3_key, force): s3_key
                for s3_key in s3_keys
            }

            for future in as_completed(future_to_key):
                s3_key = future_to_key[future]
                try:
                    track_id = future.result()
                    if track_id:
                        processed_tracks.append(track_id)
                except Exception as e:
                    print(f"❌ Error processing {s3_key}: {e}")

        return processed_tracks

    def generate_library_index(self, track_ids: List[str] = None) -> None:
        """Generate a library index file with all processed tracks"""
        library_data = {
            'generated_at': datetime.now().isoformat(),
            'source_bucket': self.source_bucket,
            'dest_bucket': self.dest_bucket,
            'dest_prefix': self.dest_prefix,
            'tracks': []
        }

        # If specific track IDs provided, use those, otherwise scan S3
        if track_ids:
            for track_id in track_ids:
                metadata_key = f"{self.dest_prefix}/{track_id}/metadata.json"
                try:
                    response = self.s3_client.get_object(Bucket=self.dest_bucket, Key=metadata_key)
                    track_data = json.loads(response['Body'].read())
                    library_data['tracks'].append(track_data)
                except ClientError as e:
                    print(f"Error reading metadata for {track_id}: {e}")

        # Save library index to S3
        library_key = f"{self.dest_prefix}/library.json"
        try:
            self.s3_client.put_object(
                Bucket=self.dest_bucket,
                Key=library_key,
                Body=json.dumps(library_data, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )
            print(f"📚 Generated library index with {len(library_data['tracks'])} tracks")
        except ClientError as e:
            print(f"❌ Failed to save library index: {e}")

# Example usage for Google Colab
def process_s3_audio_files(source_bucket: str,
                          dest_bucket: str,
                          source_prefix: str = '',
                          dest_prefix: str = 'processed_audio',
                          force_reprocess: bool = False,
                          max_workers: int = 2):
    """
    Process all audio files in an S3 bucket folder and save to another folder

    Args:
        source_bucket: S3 bucket containing source audio files
        dest_bucket: S3 bucket for processed files
        source_prefix: Prefix/folder in source bucket to process
        dest_prefix: Prefix/folder in destination bucket for results
        force_reprocess: If True, reprocess files even if they exist
        max_workers: Number of concurrent workers for processing
    """
    with S3AudioProcessor(
        source_bucket=source_bucket,
        dest_bucket=dest_bucket,
        source_prefix=source_prefix,
        dest_prefix=dest_prefix,
        max_workers=max_workers
    ) as processor:
        # List all audio files
        audio_files = processor.list_audio_files()

        if not audio_files:
            print("No audio files found to process")
            return

        print(f"Found {len(audio_files)} audio files to process")

        # Process all files
        processed_tracks = processor.process_batch(audio_files, force=force_reprocess)

        # Generate library index
        processor.generate_library_index(processed_tracks)

        print(f"🎉 Processing complete! Successfully processed {len(processed_tracks)} out of {len(audio_files)} tracks")
        print(f"📁 Results saved to: s3://{dest_bucket}/{dest_prefix}")

# Example usage (uncomment to run)
# process_s3_audio_files(
#     source_bucket="your-source-bucket",
#     dest_bucket="your-dest-bucket",
#     source_prefix="raw_audio",  # optional
#     dest_prefix="processed_audio",  # optional
#     force_reprocess=False,  # optional
#     max_workers=2  # optional
# )
