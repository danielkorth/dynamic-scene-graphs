#!/bin/bash

# Simple ZED Camera Preprocessing Script
# Usage: ./preprocess_zed.sh dataset_name
# This script expects that you have a conda environment that has the zed SDK installed

set -e  # Exit on any error

# Get dataset name from argument
DATASET_NAME="$1"
if [[ -z "$DATASET_NAME" ]]; then
    echo "Usage: $0 dataset_name"
    echo "Example: $0 game_dyn4"
    exit 1
fi

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$DATA_DIR/zed/$DATASET_NAME"
SVO_FILE="$DATA_DIR/$DATASET_NAME/$DATASET_NAME.svo2"

echo "Processing dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "SVO file: $SVO_FILE"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR/images"

# Step 1: Extract images from SVO
echo "Step 1: Extracting images from SVO..."
python3 dsg/zed/svo_export.py --mode 4 --input_svo_file "$SVO_FILE" --output_path_dir "$OUTPUT_DIR/images"

# Step 2: Create video from images
echo "Step 2: Creating video from images..."
python3 dsg/zed/merge_images_to_video_unified.py "$OUTPUT_DIR/images" -o "$OUTPUT_DIR/video.mp4" --format mp4

# Step 3: Extract camera poses
echo "Step 3: Extracting camera poses..."
conda run -n zed python3 dsg/zed/positional_tracking.py --input_svo_file "$SVO_FILE" --output_file "$OUTPUT_DIR/poses.txt"

# Step 4: Copy SVO file
echo "Step 4: Copying SVO file..."
cp "$SVO_FILE" "$OUTPUT_DIR/"

# Step 5: Undistort images (optional)
echo "Step 5: Undistorting images..."
python3 dsg/undistort.py recording="$DATASET_NAME" || echo "Undistortion failed or config missing, continuing..."

echo
echo "Processing complete!"
echo "Output saved to: $OUTPUT_DIR"


