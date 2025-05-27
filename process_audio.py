#!/usr/bin/env python3
"""
Audio Streaming Processor
Converts audio files into HLS (HTTP Live Streaming) format with multiple quality levels
for adaptive streaming applications.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Processes audio files for adaptive streaming"""
    
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
    
    def __init__(self, output_dir: str = './processed_audio', segment_duration: int = 10):
        """
        Initialize the audio processor
        
        Args:
            output_dir: Directory to store processed audio
            segment_duration: Duration of each HLS segment in seconds
        """
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if required tools are available
        self._check_dependencies()
    
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
        
        logger.info("✅ Dependencies check passed")
    
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
            logger.error(f"Failed to get audio info for {input_path}: {e}")
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
    
    def generate_track_id(self, input_path: Path, metadata: Dict) -> str:
        """
        Generate a unique track ID based on file content and metadata
        
        Args:
            input_path: Path to the audio file
            metadata: Audio metadata
            
        Returns:
            Unique track ID
        """
        # Create a hash based on file path, size, and metadata
        hasher = hashlib.md5()
        hasher.update(str(input_path).encode())
        hasher.update(str(input_path.stat().st_size).encode())
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
            
            logger.info(f"✅ Generated {quality} quality segments")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg failed for {quality} quality: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ Error processing {quality} quality: {e}")
            return False
    
    def generate_master_playlist(self, track_id: str) -> str:
        """
        Generate master playlist content for adaptive streaming
        
        Args:
            track_id: Track identifier
            
        Returns:
            Master playlist content
        """
        base_path = f"/stream/{track_id}"
        
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
    
    def save_metadata(self, track_dir: Path, track_id: str, 
                     metadata: Dict, input_path: Path) -> None:
        """
        Save track metadata to JSON file
        
        Args:
            track_dir: Track directory
            track_id: Track identifier
            metadata: Audio metadata
            input_path: Original file path
        """
        metadata_file = track_dir / 'metadata.json'
        
        track_metadata = {
            'id': track_id,
            'title': metadata['title'],
            'artist': metadata['artist'],
            'album': metadata['album'],
            'duration': int(metadata['duration']),
            'original_file': str(input_path),
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
        
        logger.info(f"💾 Saved metadata for {track_id}")
    
    def process_single_file(self, input_path: Path) -> Optional[str]:
        """
        Process a single audio file into HLS format
        
        Args:
            input_path: Path to the audio file
            
        Returns:
            Track ID if successful, None otherwise
        """
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"⚠️  Unsupported format: {input_path}")
            return None
        
        logger.info(f"🎵 Processing: {input_path}")
        
        try:
            # Get audio information
            metadata = self.get_audio_info(input_path)
            track_id = self.generate_track_id(input_path, metadata)
            
            # Create track directory
            track_dir = self.output_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each quality level
            success_count = 0
            for quality, config in self.QUALITY_CONFIGS.items():
                quality_dir = track_dir / quality
                if self.process_quality(input_path, quality_dir, quality, config):
                    success_count += 1
            
            if success_count == 0:
                logger.error(f"❌ Failed to process any quality for {input_path}")
                return None
            
            # Generate master playlist
            master_playlist_path = track_dir / 'playlist.m3u8'
            master_content = self.generate_master_playlist(track_id)
            
            with open(master_playlist_path, 'w') as f:
                f.write(master_content)
            
            # Save metadata
            self.save_metadata(track_dir, track_id, metadata, input_path)
            
            logger.info(f"✅ Successfully processed: {metadata['title']} ({track_id})")
            return track_id
            
        except Exception as e:
            logger.error(f"❌ Error processing {input_path}: {e}")
            return None
    
    def process_directory(self, input_dir: Path, max_workers: int = 4) -> List[str]:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Directory containing audio files
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of successfully processed track IDs
        """
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Directory does not exist: {input_dir}")
        
        # Find all audio files
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
            audio_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"📁 Found {len(audio_files)} audio files to process")
        
        # Process files concurrently
        processed_tracks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path
                for file_path in audio_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    track_id = future.result()
                    if track_id:
                        processed_tracks.append(track_id)
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
        
        return processed_tracks
    
    def generate_library_index(self) -> None:
        """Generate a library index file with all processed tracks"""
        library_data = {
            'generated_at': datetime.now().isoformat(),
            'tracks': []
        }
        
        # Scan all track directories
        for track_dir in self.output_dir.iterdir():
            if not track_dir.is_dir():
                continue
                
            metadata_file = track_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        track_data = json.load(f)
                        library_data['tracks'].append(track_data)
                except Exception as e:
                    logger.error(f"Error reading metadata for {track_dir}: {e}")
        
        # Save library index
        library_file = self.output_dir / 'library.json'
        with open(library_file, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📚 Generated library index with {len(library_data['tracks'])} tracks")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process audio files for adaptive streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f song.mp3                    # Process single file
  %(prog)s -d /path/to/music              # Process directory
  %(prog)s -d /music -o /processed -w 8   # Process with 8 workers
  %(prog)s -f song.mp3 --segment-duration 6  # Use 6-second segments
        """
    )
    
    parser.add_argument('-f', '--file', type=Path,
                       help='Process a single audio file')
    parser.add_argument('-d', '--directory', type=Path,
                       help='Process all audio files in directory')
    parser.add_argument('-o', '--output', type=Path, default='./processed_audio',
                       help='Output directory (default: ./processed_audio)')
    parser.add_argument('-w', '--workers', type=int, default=4,
                       help='Number of concurrent workers (default: 4)')
    parser.add_argument('--segment-duration', type=int, default=10,
                       help='HLS segment duration in seconds (default: 10)')
    parser.add_argument('--generate-index', action='store_true',
                       help='Generate library index after processing')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.file and not args.directory:
        parser.error("Must specify either --file or --directory")
    
    try:
        # Initialize processor
        processor = AudioProcessor(
            output_dir=str(args.output),
            segment_duration=args.segment_duration
        )
        
        processed_tracks = []
        
        # Process single file or directory
        if args.file:
            track_id = processor.process_single_file(args.file)
            if track_id:
                processed_tracks.append(track_id)
        
        if args.directory:
            tracks = processor.process_directory(args.directory, args.workers)
            processed_tracks.extend(tracks)
        
        # Generate library index if requested
        if args.generate_index or args.directory:
            processor.generate_library_index()
        
        logger.info(f"🎉 Processing complete! Processed {len(processed_tracks)} tracks")
        logger.info(f"📁 Output directory: {args.output}")
        
        if processed_tracks:
            print("\n✅ Successfully processed tracks:")
            for track_id in processed_tracks:
                print(f"  - {track_id}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()