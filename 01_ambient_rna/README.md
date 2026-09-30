# Ambient RNA removal (CellBender)

`cell_bender_v2.py` is the current version, and `cell_bender.py` is the previous one. The
script finds every 10x library folder under `base_path` and runs `cellbender remove-background`
on each one (CUDA, 150 epochs, FPR 0.125, expected cells capped at 13,000 and total droplets
capped at 30,000 to avoid running out of memory).

CellBender needs its own environment (the original one was the conda env `cellbender_final`,
Python 3.9):

```bash
conda create -n cellbender python=3.9 && conda activate cellbender
pip install cellbender scanpy torch
python cell_bender_v2.py
```

Edit `base_path` in the `USER SETTINGS` block, or run `tools/set_paths.sh`.
