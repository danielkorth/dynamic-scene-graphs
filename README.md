<div align="center">

# Dynamic 3D Scene Graphs from RGB-D
[Daniel Korth](https://danielkorth.io/)<sup>1,2</sup>, [Xavi Anadon](https://x.com/XaviXva)<sup>2,3</sup> <br>
<sup>1</sup>ETH Zurich, <sup>2</sup>Technical University of Munich, <sup>3</sup>University of Zaragoza

[Project Page](https://danielkorth.github.io/dynamic-scene-graphs/) | [Video](https://youtu.be/tMiMO2Wnj8Q)

https://github.com/user-attachments/assets/0115fc38-ac2c-45b3-94f3-728af601169e

</div>

Work done during 2 month ETH SSRF / RSL internship. Advised by Zuria Bauer and Daniel Barath.

**tl;dr**: ZED RGB-D recording -> SAM2 Video Tracking-> Lift Mask + Features to 3D -> Scene Graph.

## Installation

```bash
conda create -n dsg python=3.10
conda activate dsg
pip install -e .

# installing sam2
cd sam2
pip install -e .

# download sam2 checkpoints
sh scripts/download_sam.sh
```

## Configuration

We use [Hydra](https://hydra.cc/) for configuration management.

Before running any scripts, you need to configure the paths and dataset settings:

### Set Environment Variable

Set the `PROJECT_ROOT` environment variable to point to your project directory:
```bash
export PROJECT_ROOT=/path/to/your/dsg/project
```

Or add it to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
echo "export PROJECT_ROOT=/path/to/your/dsg/project" >> ~/.bashrc
source ~/.bashrc
```

configure paths in `configs/paths/default.yaml` if necessary.

### Directory Structure Setup

Ensure your data follows this structure (by default like this if you follow the `scripts/extract_zed.sh` script):
```
data/
  zed/
    your_recording_name/
      images/                    # Original RGB images
      poses.txt                  # Camera poses
      images_undistorted_crop/   # Undistorted RGB + depth images (after undistortion)
        left000000.png          # Undistorted left camera images
        left000001.png
        ...
        leftXXXXXX.png
        depth000000.png         # Undistorted depth images
        depth000001.png
        ...
        depthXXXXXX.png
        intrinsics.txt           # Camera intrinsics (after undistortion)
```

## Run (starting from RGB-D)

1. **Run SAM2 multitrack segmentation:**
   ```bash
   # Process every 10th frame with max 100 frames
   python dsg/video_tracking.py recording=<recording_name> subsample=10 max_frames=100
   ```
   check `configs/video_tracking.yaml` for all possible configurations.

2. **Visualize and build scene graph:**
   ```bash
   # Basic visualization
   python dsg/visualize_rerun.py recording=<recording_name>

   # Advanced visualization with graph updates
   python dsg/visualize_rerun_teaser.py recording=<recording_name>

   # Text-based feature retrieval
   python dsg/rerun_text_retrieval.py recording=<recording_name>
   ``` 

## Run (starting from ZED recording)

1. **Record data with ZED camera** and save as `.svo2` file
   Creates `data/zed/{recording_name}/images/` and `poses.txt`
2. **Extract frames and poses:**
   ```bash
   bash scripts/extract_zed.sh
   ```
3. **Follow steps above.**
