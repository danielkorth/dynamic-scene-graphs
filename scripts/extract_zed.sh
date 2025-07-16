#!/bin/bash

FILE_NAME="short"
INPUT_SVO_FILE="data/svo/$FILE_NAME.svo2"
OUTPUT_PATH_DIR="data/zed/$FILE_NAME"


# Video Extraction:
mkdir -p $OUTPUT_PATH_DIR/images
python src/zed/svo_export.py --mode 4 --input_svo_file $INPUT_SVO_FILE --output_path_dir $OUTPUT_PATH_DIR/images

# Video Creation:
python src/zed/merge_images_to_video_unified.py $OUTPUT_PATH_DIR/images -o $OUTPUT_PATH_DIR/video.mp4 --format mp4

# # Tracking: 
python src/zed/positional_tracking.py --input_svo_file $INPUT_SVO_FILE --output_file $OUTPUT_PATH_DIR/poses.txt
