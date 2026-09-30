"""scVI-denoised expression (the v11 feature), averaged per donor and CellType.

Same model and call as notebook (11): SCVI.load("model_scvi_adata_reduced_epochs_300_with_sex.h5ad", adata),
get_normalized_expression(library_size=1e4), then mean over the donor's cells of that cell type.
All 2000 HVGs are kept here; the v11 "top 1000 genes by mean" filter is applied later (it is label-free).
Output: data/scvi_denoised_ct.npz with, per cell type, the donor ids, n_cells and the donor x gene matrix.
"""
import time, numpy as np, pandas as pd, scanpy as sc, scvi, torch
from scvi.model import SCVI

H5 = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
MODEL = "/home/nlapique/Blood/model_scvi_adata_reduced_epochs_300_with_sex.h5ad"
OUT = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/scvi_denoised_ct.npz"
t0 = time.time()
scvi.settings.seed = 0
adata = sc.read_h5ad(H5)
model = SCVI.load(MODEL, adata)
print(f"loaded {adata.shape} + model in {time.time()-t0:.0f}s", flush=True)

pid = adata.obs["patient_id"].astype(str).str.strip().to_numpy()
ct = adata.obs["CellType"].astype(str).to_numpy()
out = {"genes": np.asarray(adata.var_names)}
for c in sorted(set(ct)):
    idx = np.where(ct == c)[0]
    X = model.get_normalized_expression(adata[idx], library_size=1e4, return_numpy=True, batch_size=4096)
    codes, donors = pd.factorize(pid[idx], sort=True)
    n = np.bincount(codes)
    S = np.zeros((len(donors), X.shape[1]), np.float64)
    np.add.at(S, codes, X)
    key = c.replace(" ", "_")
    out[f"{key}__donors"] = np.asarray(donors)
    out[f"{key}__n"] = n
    out[f"{key}__mean"] = (S / n[:, None]).astype(np.float32)
    print(f"  {c}: {len(idx)} cells, {len(donors)} donors  {time.time()-t0:.0f}s", flush=True)
    del X; torch.cuda.empty_cache()
np.savez_compressed(OUT, **out)
print(f"done {time.time()-t0:.0f}s", flush=True)
