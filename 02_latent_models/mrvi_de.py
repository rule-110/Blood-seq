#!/usr/bin/env python
import scvi
import scanpy as sc
import scipy.sparse as sp
import pandas as pd
from scvi.external import MRVI
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
from scvi.external import MRVI
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(15)

import scanpy as sc
from scvi.external.mrvi_torch import TorchMRVI

print("Loading data...")
test_adata = sc.read_h5ad("/home/nlapique/Blood/adata_hvg3000_remove_low_cellsdpnnors.h5ad")

print("Loading model...")
model = TorchMRVI.load(
    "/home/nlapique/Blood/mrvi_blood_model_3000_hvg_donnors_men_500epochs",
    test_adata,
    accelerator="cpu"
)

print("Preparing...")
model.sample_info["VOmax26_bin_male"] = model.sample_info["VOmax26_bin_male"].astype("category")

print("Running DE...")
de_res = model.differential_expression(
    sample_cov_keys=["VOmax26_bin_male"],
    store_lfc=True,
    mc_samples=40,
    use_vmap=False
)

print("Post-processing gene-level results...")

# -----------------------------
# 1. Gene names
# -----------------------------
genes = de_res["gene"].values  # (n_genes,)

# -----------------------------
# 2. Extract LFC: (cells, 1, genes) -> (cells, genes)
# -----------------------------
lfc = np.squeeze(de_res["lfc"].values, axis=1)

# -----------------------------
# 3. Gene-level summaries
# -----------------------------
gene_lfc = np.mean(lfc, axis=0)
effect_size = np.mean(np.abs(lfc), axis=0)

# -----------------------------
# 4. Gene-level p-values
#     (most significant cell per gene)
# -----------------------------
pval = np.squeeze(de_res["pvalue"].values, axis=1)
padj = np.squeeze(de_res["padj"].values, axis=1)

gene_pval = np.nanmin(pval, axis=0)
gene_padj = np.nanmin(padj, axis=0)

# -----------------------------
# 5. Final gene-level table
# -----------------------------
de_gene = pd.DataFrame({
    "gene": genes,
    "lfc": gene_lfc,
    "effect_size": effect_size,
    "pvalue": gene_pval,
    "padj": gene_padj
})

de_gene = de_gene.sort_values("pvalue")

# -----------------------------
# 6. Save (SAFE)
# -----------------------------
out = "/home/nlapique/Blood/de_results_VOmax_gene_level.csv"
de_gene.to_csv(out, index=False)

print(f"✓ Saved gene-level DE results to:\n{out}")
print(de_gene.head(100))