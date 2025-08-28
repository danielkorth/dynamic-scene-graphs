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

# download sam2 checkpoints
sh scripts/download_sam.sh
```

## Configuration

We use [Hydra](https://hydra.cc/) for configuration management.

Before running any scripts, you need to configure the paths and dataset settings:

### 1. Set Environment Variable

Set the `PROJECT_ROOT` environment variable to point to your project directory:
```bash
export PROJECT_ROOT=/path/to/your/dsg2/project
```

Or add it to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
echo "export PROJECT_ROOT=/path/to/your/dsg2/project" >> ~/.bashrc
source ~/.bashrc
```

### 2. Configure Paths

The paths are defined in `configs/paths/default.yaml`. The default configuration uses:
- `root_dir`: Points to your project root (via `$PROJECT_ROOT` env var)
- `data_dir`: `${root_dir}/data/` - Where your datasets are stored
- `output_dir`: Hydra's runtime output directory (automatically managed)

### 3. Dataset-Specific Configuration

For each dataset/recording, you need to set these values in `configs/video_tracking.yaml`:

#### **Required Manual Configuration:**
- **`recording`**: Name of your dataset (e.g., `"umbrella2"`, `"kitchen_scene"`)
- **`source_folder`**: Dataset root directory (automatically derived from `recording`)
- **`images_folder`**: Path to undistorted RGB images (automatically derived)
- **`intrinsics_file`**: Path to camera intrinsics file (automatically derived)

#### **Example Configuration:**
```yaml
# Dataset identifier
recording: my_custom_recording

# These paths are automatically derived:
source_folder: ${paths.data_dir}zed/my_custom_recording
images_folder: ${paths.data_dir}zed/my_custom_recording/images_undistorted_crop
intrinsics_file: ${images_folder}/intrinsics.txt
```

### 4. Directory Structure Setup

Ensure your data follows this structure:
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

## General Workflow

1. **Run SAM2 multitrack segmentation:**
   ```bash
   # Process every 10th frame with max 100 frames
   python dsg/video_tracking.py recording=<recording_name> subsample=10 max_frames=100
   ```

2. **Visualize and build scene graph:**
   ```bash
   # Basic visualization
   python dsg/visualize_rerun.py recording=<recording_name>

   # Advanced visualization with graph updates
   python dsg/visualize_rerun_teaser.py recording=<recording_name>

   # Text-based feature retrieval
   python dsg/rerun_text_retrieval.py recording=<recording_name>
   ``` 
## Workflow with ZED camera

1. **Record data with ZED camera** and save as .svo2 file

2. **Extract frames and poses:**
   ```bash
   sh scripts/extract_zed.sh
   ```
   Creates `data/zed/{recording_name}/images/` and `poses.txt`

3. **Undistort the images:**
   ```bash
   # Basic undistortion (no cropping)
   python dsg/undistort.py recording=<recording_name>

   # Undistortion with automatic cropping (removes black regions)
   python dsg/undistort.py recording=<recording_name> undistort=true
   ```

-> General workflow
