#!/bin/bash

# Create output directory
mkdir -p data/living_room_1
cd data/living_room_1

# Download files (update URLs if needed)
wget --no-check-certificate -c https://redwood-data.org/indoor/data/livingroom.ply.zip
wget --no-check-certificate -c https://redwood-data.org/indoor/data/livingroom1-color.zip
wget --no-check-certificate -c https://redwood-data.org/indoor/data/livingroom1-depth-clean.zip
wget --no-check-certificate -c https://redwood-data.org/indoor/data/livingroom1-traj.txt

# Unzip files into different subdirectories
unzip livingroom1-color.zip -d color
unzip livingroom1-depth-clean.zip -d depth
unzip livingroom.ply.zip 

rm livingroom1-color.zip
rm livingroom1-depth-clean.zip
rm livingroom.ply.zip

echo "All Living Room 1 files downloaded to $(pwd)"
