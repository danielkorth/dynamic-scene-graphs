#!/bin/bash

conda activate zed

FILE_NAMES=("office4")
for FILE_NAME in ${FILE_NAMES[@]}; do
    INPUT_SVO_FILE="data/svo/$FILE_NAME.svo2"
    OUTPUT_PATH_DIR="data/zed/$FILE_NAME"

    # Tracking: 
    python src/zed/positional_tracking.py --input_svo_file $INPUT_SVO_FILE --output_file $OUTPUT_PATH_DIR/poses.txt

    scp -r $OUTPUT_PATH_DIR/poses.txt xanadon@129.132.245.33:/local/home/xanadon/dynamic-scene-graphs/$OUTPUT_PATH_DIR/poses.txt
done
