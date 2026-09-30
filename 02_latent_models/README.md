# Latent models (scVI / MRVI)

- `mrvi_training.ipynb`: MRVI setup and training on the combined AnnData (outputs stripped).
- `mrvi_de.py`, `mrvi_de_cell_type.py`, `mrvi_scampy_de_genes.py`: MRVI differential expression
  (these were written for VO2max).

The scVI model used downstream (`model_scvi_adata_reduced_epochs_300_with_sex.h5ad/`, 300 epochs,
2000 HVGs, sex as a covariate) is loaded by `notebooks/biomarker_v11_*` with
`SCVI.load(path, adata)`.
