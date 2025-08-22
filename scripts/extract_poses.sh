#!/bin/bash

# FILE_NAMES=("office4" "office3" "office2" "office1" "umbrella1" "umbrella2" "occlusion_pen" "occlusion_ball" "ball1" "ball2" "remove1" "remove2")
FILE_NAMES=("game_dyn6")
# FILE_NAMES=("office4")
for FILE_NAME in ${FILE_NAMES[@]}; do
    INPUT_SVO_FILE="data/svo/$FILE_NAME.svo2"
    OUTPUT_PATH_DIR="data/zed/$FILE_NAME"

    rm -rf $OUTPUT_PATH_DIR
    scp -r xanadon@129.132.245.33:/local/home/xanadon/dynamic-scene-graphs/$OUTPUT_PATH_DIR $OUTPUT_PATH_DIR
    scp xanadon@129.132.245.33:/local/home/xanadon/dynamic-scene-graphs/$INPUT_SVO_FILE $OUTPUT_PATH_DIR/$FILE_NAME.svo2

    rm $OUTPUT_PATH_DIR/poses.txt
    # Tracking: 
    conda run -n zed python dsg/zed/positional_tracking.py --input_svo_file $OUTPUT_PATH_DIR/$FILE_NAME.svo2 --output_file $OUTPUT_PATH_DIR/poses.txt

    scp $OUTPUT_PATH_DIR/poses.txt xanadon@129.132.245.33:/local/home/xanadon/dynamic-scene-graphs/$OUTPUT_PATH_DIR/poses.txt
done
