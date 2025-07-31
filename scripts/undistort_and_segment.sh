#!/bin/bash

DATASET_NAMES=("ball1" "ball2" "umbrella2" "occlusion_pen" "short" "remove2" "office4")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    python src/undistort.py recording=$DATASET_NAME
    python src/sam2_reinit.py recording=$DATASET_NAME
done