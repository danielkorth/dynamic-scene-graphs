<div align="center">

# Dynamic 3D Scene Graphs from RGB-D
[Daniel Korth](https://danielkorth.io/)<sup>1</sup>, [Xavi Anadon](https://x.com/XaviXva)<sup>2</sup> <br>
<sup>1</sup>Technical University of Munich, <sup>2</sup>University of Zaragoza

[Project Page](https://danielkorth.github.io/dynamic-scene-graphs/) | [Video](https://youtu.be/tMiMO2Wnj8Q)

https://github.com/user-attachments/assets/0115fc38-ac2c-45b3-94f3-728af601169e

</div>



Work done during 2 month ETH SSRF / RSL internship. Advised by Zuria Bauer and Daniel Barath.

tl;dr: ZED RGB-D recording -> SAM2 Video Tracking-> Lift Mask + Features to 3D -> Scene Graph.

## Setup

```bash
conda create -n dsg python=3.10
conda activate dsg
pip install -e .

# installing sam2
cd sam2
pip install -e .
```

## Data Structure
The project expects the following folder structure for data:

```bash
# Download example dataset
sh scripts/download_redwood.sh
```

For ZED camera recordings, the expected structure is:
```
data/
  zed/
    {recording_name}/
      images/                    # Original RGB images
      poses.txt                  # Camera poses
      intrinsics.txt             # Camera intrinsics
      images_undistorted_crop/   # Undistorted RGB images (after step 3)
      masks_{stride}_{max_frames}/     # SAM2 segmentation masks
      mask_images_{stride}_{max_frames}/ # Visualization of masks
      obj_points_history/        # 3D object point clouds
```

## Workflow

1. **Record data with ZED camera** and save as .svo2 file

2. **Extract frames and poses:**
   ```bash
   sh scripts/extract_zed.sh
   ```
   Creates `data/zed/{recording_name}/images/` and `poses.txt`

3. **Undistort the images:**
   ```bash
   # Basic undistortion (no cropping)
   python src/undistort.py recording=<recording_name>

   # Undistortion with automatic cropping (removes black regions)
   python src/undistort.py recording=<recording_name> undistort=true
   ```

4. **Run SAM2 multitrack segmentation:**
   ```bash
   # Process every 10th frame with max 100 frames
   python src/sam2_tracking.py recording=<recording_name> stride=10 max_frames=100 sam=tiny
   ```

5. **Visualize and build scene graph:**
   ```bash
   python src/visualize_rerun.py
   ``` 
