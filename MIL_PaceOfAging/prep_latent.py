"""scVI latent (per cell) for the MIL experiments: same model as notebook (11)."""
import numpy as np, pandas as pd, scanpy as sc, scvi
from scvi.model import SCVI
scvi.settings.seed = 0
ad = sc.read_h5ad("/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad")
m = SCVI.load("/home/nlapique/Blood/model_scvi_adata_reduced_epochs_300_with_sex.h5ad", ad)
Z = m.get_latent_representation(batch_size=4096).astype(np.float32)
np.save("latent_scvi.npy", Z)
pd.DataFrame({"patient_id": ad.obs["patient_id"].astype(str).str.strip().values,
              "CellType": ad.obs["CellType"].astype(str).values}).to_parquet("cells.parquet")
print(Z.shape)
