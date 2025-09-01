<div align="center">

# Dynamic 3D Scene Graphs from RGB-D
[Daniel Korth](https://danielkorth.io/)<sup>1,2,\*</sup>, [Xavier Anadon](https://x.com/XaviXva)<sup>1,3,\*</sup>, [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)<sup>1,4</sup>, [Zuria Bauer](https://zuriabauer.com)<sup>1</sup>, [Daniel Barath](https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath)<sup>1,5</sup> <br>
<sup>1</sup>ETH Zurich, <sup>2</sup>Technical University of Munich, <sup>3</sup>University of Zaragoza, <sup>4</sup>Microsoft, <sup>5</sup>HUN-REN SZTAKI <br>
<sup>*</sup>Equal contribution

[Project Page](https://danielkorth.github.io/dynamic-scene-graphs/) | [Video](https://youtu.be/tMiMO2Wnj8Q)

https://github.com/user-attachments/assets/88b1082f-08dd-4f0e-bcb5-6892fbf17200

</div>

Work done during 2-month ETH Summer Research Fellowship (SSRF & RSF). Advised by Zuria Bauer and Daniel Barath.

**tl;dr**: RGB-D recording + camera poses -> SAM2 Video Tracking -> Lift Mask + Features to 3D -> Scene Graph.

Please check the [project page](https://danielkorth.github.io/dynamic-scene-graphs/) for more details.

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

### Directory Structure Setup

Our data structure follows the ZED extraction scripts, but you can use your own RGB-D data. If using different formats, adjust the paths in `configs/paths/default.yaml` and `configs/video_tracking.yaml`.

Default structure (from ZED extraction):
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

## Run (starting from existing RGB-D + poses)

If you already have RGB-D images and camera poses:

1. **Run SAM2 multitrack segmentation:**
   ```bash
   # Process every 10th frame with max 100 frames
   python dsg/video_tracking.py recording=<recording_name> subsample=10 max_frames=100
   ```
   Check `configs/video_tracking.yaml` for all configurations.

2. **Visualize and build scene graph:**
   ```bash
   # Basic visualization
   python dsg/viz_rerun.py recording=<recording_name>

   # Advanced visualization with graph updates
   python dsg/viz_rerun_teaser.py recording=<recording_name>

   # Text-based feature retrieval
   python dsg/viz_clip_similarity.py recording=<recording_name>

   # Object reconstruction
   python dsg/viz_obj_reconstruction.py recording=<recording_name>
   ```

## Run (starting from ZED recording)

If you have a raw ZED recording:

1. **Record data with ZED Mini camera** and save as `.svo2` file
2. **Extract frames and poses:**
   ```bash
   bash scripts/extract_zed.sh
   ```
3. **Follow steps above.**

## Acknowledgments

Our work builds heavily on foundations models such as [SAM](https://sam2.metademolab.com/) and [CLIP](https://openai.com/blog/clip/) and [SALAD](https://github.com/serizba/salad). We thank the authors for their work and open-source code.

## Citing

```bibtex
@article{korth2025dynamic,
  author    = {Korth, Daniel and Anadon, Xavier and Pollefeys, Marc and Bauer, Zuria and Barath, Daniel},
  title     = {Dynamic 3D Scene Graphs from RGB-D},
  year      = {2025},
}
```
