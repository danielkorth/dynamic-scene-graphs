# dynamic-scene-graphs

## Setup


```bash
conda create -n dsg python=3.10
conda activate dsg
pip install uv 
uv pip install -e .
# installing sam2
cd sam2
uv pip install -e .
```

## Data Structure
The project expects the following folder structure for data:

```bash
sh scripts/download_redwood.sh
```

which creates the following structure:
```
data/
  living_room_1/
    color/
    depth/
    livingroom.ply
    livingroom1-traj.txt
```

## Workflow

Understanding the workflow:
1. Record data and save a .svo2 file
2. scripts/extract_zed.sh -> create data/zed/{file_name}/images/ and data/zed/{file_name}/poses.txt
    -> I need the correct intrinsics and distortion coefficients in this folder!!!
3. **Undistort the images:**
   ```bash
   # Basic undistortion (no cropping)
   python src/undistort.py recording=<file_name>
   
   # Undistortion with automatic cropping (removes black regions)
   python src/undistort.py recording=<file_name> undistort=true
   ```

4. **Run SAM2 multitrack segmentation:**
```bash
   # Process every 5th frame with max 100 frames
   python src/sam2_tracking.py recording=<file_name> stride=5 max_frames=100 sam=tiny
```

5. run visualize_rerun.py 