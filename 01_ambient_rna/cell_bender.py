#!/usr/bin/env python
import os
import glob
import subprocess
import scanpy as sc
import numpy as np
import sys
import torch


# Check GPU availability
num_gpus = torch.cuda.device_count()
print(f"PyTorch sees {num_gpus} GPU(s) available.")
for i in range(num_gpus):
    print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

# ====== USER SETTINGS ======
base_path = "/home/nlapique/Blood/cell_bender3/"      # Path containing all library folders
# We now use the current python interpreter to call cellbender as a module
# This ensures it uses the Python 3.9 version you just set up.
use_cuda = True                      
umi_threshold = 500                 
total_droplet_multiplier = 4         
epochs = 150
fpr = 0.125
# ========= cbv$===================

# Detect all libraries
libraries = glob.glob(os.path.join(base_path, "*/raw_feature_bc_matrix"))
if not libraries:
    raise ValueError(f"No libraries found in {base_path}!")

print(f"Found {len(libraries)} libraries:")
for lib in libraries:
    print(f" - {lib}")

# Process each library
for lib in libraries:
    print(f"\n===== Processing {lib} =====")

    # Estimate expected cells automatically
    print("Estimating expected cells using UMI threshold...")
    # Load data (cache=True helps if you run this multiple times)
    adata = sc.read_10x_mtx(lib, cache=False)
    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    expected_cells = int(np.sum(total_counts > umi_threshold))
    total_droplets = expected_cells * total_droplet_multiplier

    print(f"Estimated cells: {expected_cells}")
    print(f"Total droplets included: {total_droplets}")

    # Build CellBender command
    # We use 'sys.executable -m cellbender' to force the use of the 3.9 environment
    # Build CellBender command
    
    # This finds the 'cellbender' command that belongs specifically to your conda env
    conda_bin_dir = os.path.dirname(sys.executable)
    cellbender_path = os.path.join(conda_bin_dir, "cellbender")

    # Build CellBender command
    output_file = os.path.join(os.path.dirname(lib), "cellbender_output.h5")
    cmd = [
        "cellbender", 
        "remove-background",
        f"--input={lib}",
        f"--output={output_file}",
        f"--expected-cells={expected_cells}",
        f"--total-droplets-included={total_droplets}",
        f"--epochs={epochs}",
        f"--fpr={fpr}"
        # DO NOT put "--cuda" here if you are using the 'if use_cuda' block below
    ]
    if use_cuda:
        cmd.append("--cuda")
    print("Running CellBender...")
    print(" ".join(cmd))  
    
    try:
        # We use shell=False for better security/stability with list commands
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {lib}: {e}")
        continue

print("\nAll libraries processed successfully!")
