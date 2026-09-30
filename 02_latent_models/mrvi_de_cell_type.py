#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(15)

import scanpy as sc
import pandas as pd
import numpy as np
from scvi.external.mrvi_torch import TorchMRVI

print("Loading data...")
test_adata = sc.read_h5ad("/home/nlapique/Blood/adata_hvg3000_remove_low_cellsdpnnors.h5ad")

# Get cell types
cell_types = test_adata.obs["CellType"].unique()
print(f"Found {len(cell_types)} cell types")

all_results = []

for ct in cell_types:
    print(f"\n{'='*60}")
    print(f"Cell type: {ct}")
    print(f"{'='*60}")
    
    # Subset to cell type
    adata_ct = test_adata[test_adata.obs["CellType"] == ct].copy()
    adata_ct.X = adata_ct.X.tocsr()
    
    print(f"  Cells: {len(adata_ct)}")
    
    # Load model for this cell type
    print(f"  Loading model...")
    model = TorchMRVI.load(
        "/home/nlapique/Blood/mrvi_blood_model_3000_hvg_donnors_men_500epochs",
        adata_ct,
        accelerator="cpu"
    )
    
    # Prepare
    model.sample_info["VOmax26_bin_male"] = (
        model.sample_info["VOmax26_bin_male"].astype("category")
    )
    
    # Check groups
    groups = model.sample_info["VOmax26_bin_male"].value_counts()
    print(f"  Groups: {dict(groups)}")
    
    if len(groups) < 2:
        print(f"  ✗ Skipping: only {len(groups)} group(s)")
        del model
        continue
    
    # Run MRVI DE
    try:
        print(f"  Running MRVI DE...")
        de_res = model.differential_expression(
            sample_cov_keys=["VOmax26_bin_male"],
            store_lfc=True,
            mc_samples=20,  # Lower for speed
            use_vmap=False
        )
        
        # Extract gene-level results (your aggregation code)
        genes = de_res["gene"].values
        lfc = np.squeeze(de_res["lfc"].values, axis=1)
        gene_lfc = np.mean(lfc, axis=0)
        effect_size = np.mean(np.abs(lfc), axis=0)
        
        pval = np.squeeze(de_res["pvalue"].values, axis=1)
        padj = np.squeeze(de_res["padj"].values, axis=1)
        gene_pval = np.nanmin(pval, axis=0)
        gene_padj = np.nanmin(padj, axis=0)
        
        # Create DataFrame
        de_ct = pd.DataFrame({
            "cell_type": ct,
            "gene": genes,
            "lfc": gene_lfc,
            "effect_size": effect_size,
            "pvalue": gene_pval,
            "padj": gene_padj
        })
        
        sig_genes = (de_ct['padj'] < 0.05).sum()
        print(f"  ✓ Complete: {sig_genes} significant genes")
        
        # Save
        ct_safe = ct.replace(" ", "_").replace("/", "_")
        de_ct.to_csv(f"/home/nlapique/Blood/de_VOmax_{ct_safe}.csv", index=False)
        
        all_results.append(de_ct)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

# Combine
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("/home/nlapique/Blood/de_VOmax_all_celltypes_mrvi.csv", index=False)
    print(f"\n✓ Saved {len(combined)} results")
else:
    print("\n✗ No results!")