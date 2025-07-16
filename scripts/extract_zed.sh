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

scp -r $OUTPUT_PATH_DIR xanadon@129.132.245.33:/local/home/xanadon/dynamic-scene-graphs/$OUTPUT_PATH_DIR
# scp -r $OUTPUT_PATH_DIR dkorth@129.132.245.22:/local/home/dkorth/Projects/dynamic-scene-graphs/data/zed
