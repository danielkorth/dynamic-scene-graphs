#!/usr/bin/env bash

# Usage: copy_and_run_sam2_multitrack.sh <source_folder> <output_folder> [--stride N] [--max-frames M]

set -e

SOURCE_FOLDER="$1"
OUTPUT_FOLDER="$2"
TMP_FOLDER="$OUTPUT_FOLDER/tmp"
STRIDE=1
MAX_FRAMES=0

# Parse optional arguments
while [[ $# -gt 2 ]]; do
  case "$3" in
    --stride)
      STRIDE="$4"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$4"
      shift 2
      ;;
    *)
      echo "Unknown option: $3"
      exit 1
      ;;
  esac
done

if [ -z "$SOURCE_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ]; then
  echo "Usage: $0 <source_folder> <output_folder> [--stride N] [--max-frames M]"
  exit 1
fi

# If TMP_FOLDER exists, remove all its contents
if [ -d "$TMP_FOLDER" ]; then
  rm -rf "$TMP_FOLDER"/*
fi

mkdir -p "$TMP_FOLDER"

# Copy left*.png images from the images subdirectory to tmp folder
find "$SOURCE_FOLDER/images" -maxdepth 1 -type f -name 'left*.png' -exec cp {} "$TMP_FOLDER" \;

# Select and convert using Python (stride and max-frames)
python scripts/select_and_convert_frames.py --input "$TMP_FOLDER" --stride "$STRIDE" --max-frames "$MAX_FRAMES"

# Ensure all .png files are removed from tmp folder
rm -f "$TMP_FOLDER"/*.png

# Run the python script
python scripts/sam2_multitrack.py --source_frames "$TMP_FOLDER" --output_dir "$OUTPUT_FOLDER"

# Remove the tmp folder after processing
rm -rf "$TMP_FOLDER" 