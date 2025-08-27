#!/bin/bash

# --------- USER CONFIGURATION ---------
IMAGES_DIR="/local/home/dkorth/Projects/dynamic-scene-graphs/data/colmap/images"           # Folder with your images
WORKSPACE="/local/home/dkorth/Projects/dynamic-scene-graphs/data/colmap"         # Output workspace folder
DATABASE_PATH="$WORKSPACE/database.db"      # SQLite database file
SPARSE_DIR="$WORKSPACE/sparse"              # Output for sparse reconstruction
DENSE_DIR="$WORKSPACE/dense"                # Output for dense reconstruction
# --------------------------------------

mkdir -p "$WORKSPACE"
mkdir -p "$SPARSE_DIR"
mkdir -p "$DENSE_DIR"

# 1. Feature extraction
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_model "PINHOLE" \
    --ImageReader.camera_params "437.72, 428.63, 532.4, 283.88"

# 2. Feature matching (exhaustive for small/medium datasets)
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

# 3. Sparse reconstruction (Structure-from-Motion)
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"
