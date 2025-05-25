#!/bin/bash

# Usage: ./process-audio.sh input_file track_id
# Example: ./process-audio.sh uploads/song.mp3 track1

INPUT_FILE=$1
TRACK_ID=$2

if [ -z "$INPUT_FILE" ] || [ -z "$TRACK_ID" ]; then
    echo "Usage: $0 <input_file> <track_id>"
    exit 1
fi

# Create output directories
mkdir -p "audio/$TRACK_ID/low"
mkdir -p "audio/$TRACK_ID/med" 
mkdir -p "audio/$TRACK_ID/high"

echo "Processing $INPUT_FILE -> $TRACK_ID..."

# Generate segments for different qualities
ffmpeg -i "$INPUT_FILE" \
    -map 0:a -c:a libopus -b:a 32k -f segment -segment_time 10 \
    -segment_list "audio/$TRACK_ID/low/playlist.m3u8" \
    -segment_list_flags +live \
    "audio/$TRACK_ID/low/segment_%03d.webm" \
    \
    -map 0:a -c:a libopus -b:a 64k -f segment -segment_time 10 \
    -segment_list "audio/$TRACK_ID/med/playlist.m3u8" \
    -segment_list_flags +live \
    "audio/$TRACK_ID/med/segment_%03d.webm" \
    \
    -map 0:a -c:a libopus -b:a 96k -f segment -segment_time 10 \
    -segment_list "audio/$TRACK_ID/high/playlist.m3u8" \
    -segment_list_flags +live \
    "audio/$TRACK_ID/high/segment_%03d.webm"

echo "✅ Processed $TRACK_ID successfully!"