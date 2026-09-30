# Data

Nothing in this file list is in the repository. Access goes through the Dunedin Study
data-access process.

| file (relative to the data root) | content |
|---|---|
| `adata_reduced_HVG_2000_with_sex_qcmt.h5ad` | ~2.4 GB. 2000 HVGs, raw counts in `layers["counts"]`, `obs` with `patient_id`, `CellType`, `sex`, `QC_MT.pct`, `total_counts` and trait columns |
| `cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad` | ~13 GB. Full transcriptome (29,572 genes) after CellBender, plus 23 DunedinPACE-related columns |
| `model_scvi_adata_reduced_epochs_300_with_sex.h5ad/` | trained scVI model (a **directory**, despite the suffix) used by the v11 notebook, MIL and the permutation test |
| `mrvi_*` model directories | MRVI runs (all donors / men only / CellBender-corrected / ensemble seeds) |
| `BroadInst_DunedinPheno_20240829.txt` | phenotype table, one row per participant, key `snum` (= `obs['patient_id']`) |

## Known quirks

- The `cellbender_final` conda env's `anndata` cannot `read_h5ad` the 13 GB file
  (`IORegistryError` on `/uns/log1p`). The scripts read it with `h5py` + `read_elem` instead.
- `cortis26np` is missing from the HVG h5ad. The no-scVI notebook merges it in from the phenotype txt.
- `crp26ns` is in whole mg/L, so its tertiles are uneven. `PackYrsBy26` is 0 for all non-smokers.
  The tertile code uses Low = non-smokers and median-splits the rest.
- Pickles under `NIGHT_PaceOfAging_snRNA/data/` were written with numpy 2 and need the
  environment in `requirements.txt`.

## Keeping participant data out of git

- `.gitignore` blocks h5ad/npy/pkl/parquet/xlsx/txt, all `data/` and cache folders, per-donor
  predictions (`results/oof/`, `LOCKBOX_predictions.csv`), per-run output folders
  (`MULTISPLIT_*`, reports) and `lockbox.csv` (donor IDs).
- **Notebook outputs can contain donor-level values.** Strip them before committing:
  `python tools/strip_notebook.py in.ipynb out.ipynb` (or install `nbstripout`).
- The only result files committed are aggregate metrics (AUC, CIs, per-experiment summaries).
