# dynamic-scene-graphs

## Setup


```bash
conda create -n dsg python=3.11
conda activate dsg
pip install uv 
uv pip install -r requirements.txt
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
