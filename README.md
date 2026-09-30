# Bloodseq: whole-blood snRNA-seq and the Pace of Aging

Analysis code for single-nucleus RNA-seq of whole blood from ~800 Dunedin Study
participants, all sampled at **age 26**. The main question is whether blood transcriptomes
at 26 predict **PaceOfAgingP45**, the rate of multi-system biomarker change from age 26 to 45.

> **No data is included.** The snRNA-seq matrices and phenotypes are covered by the Dunedin
> Study data-access agreement. See [`docs/DATA.md`](docs/DATA.md) for the files the code
> expects and the rules for keeping participant data out of this repository.

## Study design, and what counts as a fair predictor

The design is **prospective**: expression at 26 is used to predict *future* aging. The only
fair predictors or baselines are **age-26 variables** (`*26`). Later measurements
(`*32`, `*38`, `*45`), slopes and intercepts (`*_iz_snL`, `*_iz_inL`), and anything else
fit over 26–45 are future information, and pace itself is built from them. Every pipeline
here keeps to that rule.

## Pipeline

```
raw 10x libraries
  └─ 01_ambient_rna/          CellBender ambient-RNA removal, one run per library
      └─ 02_latent_models/    scVI / MRVI training + MRVI differential expression
          ├─ notebooks/       per-trait feature engineering and biomarker discovery
          │    bmi_classification_mrvi      MRVI feature blocks (composition, u-latent, local distances) + nested CV
          │    biomarker_v11_scvi_*         scVI-denoised pseudobulk → XGBoost + stability selection, SHAP, PDF/Excel reports,
          │                                 tertile sweeps, three-model comparison, clinical → snRNA cascade
          │    biomarker_v12_libcorr_proxy  full-transcriptome library-corrected pseudobulk, "proxy" features (predicted age-26 traits)
          │    biomarker_noscvi_*           the same tertile sweep with no scVI at all (raw-count pseudobulk)
          ├─ NIGHT_PaceOfAging_snRNA/  systematic model search with a held-out lock-box (main result, see below)
          ├─ MIL_PaceOfAging/          multiple-instance learning baselines (ABMIL, MixMIL, ScRAT, CloudPred, scMILD)
          ├─ PERMTEST_tertile_scvi/    donor-level random-tertile permutation test for the HDL/CRP tertile gradients
          └─ scImmuAging_clock/        sc-ImmuAging cell-type clocks (Li et al., Nat Aging 2025) mapped to gene symbols
plots/        stand-alone plotting scripts (AUC barplots, pace by HDL/CRP tertile)
tools/        set_paths.sh (relocate hardcoded paths), strip_notebook.py (remove outputs before committing)
```

## Main results (short version)

Full numbers: [`docs/RESULTS.md`](docs/RESULTS.md) and
[`NIGHT_PaceOfAging_snRNA/REPORT.md`](NIGHT_PaceOfAging_snRNA/REPORT.md).

| model (fast agers = top 20 % vs bottom 60 %) | CV AUC | lock-box AUC [95 % CI] |
|---|---|---|
| covariates only (sex, cells, UMI, % mt) | 0.55 | 0.54 [0.43–0.65] |
| snRNA-seq only, proxy model | 0.70 | **0.70 [0.58–0.80]** |
| BMI only | 0.70 | 0.72 [0.61–0.82] |
| BMI + snRNA-seq | 0.76 | 0.79 [0.70–0.88] |
| 13 age-26 clinical traits | 0.80 | 0.83 [0.74–0.90] |
| 13 clinical + snRNA-seq | 0.81 | 0.82 [0.73–0.90] |

The snRNA-seq signal is real but modest. Most of it is smoking: it disappears for
`POA45_nocal`, and the top gene is *AHRR*. It adds nothing once the age-26 clinical traits
are in the model. scVI latents, MIL models, cell-type composition (CNA) and sc-ImmuAging
clocks all do no better than plain pseudobulk.

## Setup

```bash
git clone <this repo> && cd bloodseq-pace-of-aging
conda env create -f environment.yml && conda activate bloodseq   # or: python -m venv .venv && pip install -r requirements.txt
# The code uses absolute paths from the original workstation. Rewrite them once:
tools/set_paths.sh /path/to/data_root /path/to/BroadInst_DunedinPheno_20240829.txt
```

`set_paths.sh` maps `/home/nlapique/Blood/script/` to this repo,
`/home/nlapique/Blood/` to your data root, and the phenotype file path to yours. Commit
or stash first, so you can review the diff.

Hardware used: one CUDA GPU (scVI, XGBoost `device="cuda"`), 64 GB+ RAM for the
full-transcriptome h5ad (13 GB on disk). CellBender runs in its own environment (see
[`01_ambient_rna/README.md`](01_ambient_rna/README.md)).

## Running the analyses

**Notebooks** (`notebooks/`, run in JupyterLab). Each has a single `GLOBAL CONFIG` cell
(`BIOMARKER`, `TARGET_GROUP`, quantile cutoffs, number of iterations). Change that cell
and run everything below it. `BIOMARKER` must match an `adata.obs` column exactly. Long
sweeps save after every combination and resume when re-run (`RESUME_*` flags).
`v11_counts_tertile_sweep.py` is run from inside the v11 notebook with `%run -i`.

**Model search with lock-box** (`NIGHT_PaceOfAging_snRNA/scripts/`):

```bash
cd NIGHT_PaceOfAging_snRNA/scripts
python prep.py && python prep2_bylib.py && python prep3_aux_labels.py && python prep4_lognorm_moments.py
python prep5_scimmuaging.py && python prep6_scvi_denoised.py      # only needed for exp14 / exp15
python exp01_baselines.py            # also creates data/lockbox.csv (20 % of donors, seed 2026)
for f in exp0[2-9]_*.py exp1[0-6]_*.py; do python "$f"; done   # each appends rows to results/results.csv
python rescore_cutoffs.py 0.7 0.9    # optional: rescore with other High/Low quantiles
python final_lockbox.py              # ONE-TIME scoring on the lock-box: do not use it for model selection
python final_figure_and_models.py
```

`evalkit.py` holds the rules: the same repeated 5×3 CV folds for every experiment, all
supervised steps inside the training fold, and the lock-box used only by `final_lockbox.py`.
The published lock-box has **already been used once**. Any new model selection must use
CV on the dev donors only.

**MIL** (`MIL_PaceOfAging/`): see its [README](MIL_PaceOfAging/README.md).
**Permutation test** (`PERMTEST_tertile_scvi/`):
`python permtest_tertile.py cache`, then `run 60`, then `summary`.

## Conventions

- Required covariates: `sex` and `QC_MT.pct` (donor mean, not a single cell's value).
- Sex coding in clinical tables varies: map `{'1':'F','2':'M','F':'F','M':'M'}`.
- Merge key: phenotype `snum` == `obs['patient_id']`.
- Report names: `Rapport_{biomarker}_{cell_type}_{group}.pdf`,
  `Tableau_{biomarker}_{cell_type}_{group}.csv`, `RESULTATS_*_{BIOMARKER}.xlsx`.
  These outputs are git-ignored.

## Third-party code

- sc-ImmuAging clock coefficients: derived from
  [CiiM-Bioinformatics-group/scImmuAging](https://github.com/CiiM-Bioinformatics-group/scImmuAging)
  (Apache-2.0); see [`scImmuAging_clock/README.md`](scImmuAging_clock/README.md).
- The MIL wrappers import the official MixMIL, ScRAT, CloudPred and scMILD code, which is not
  vendored here; see [`MIL_PaceOfAging/README.md`](MIL_PaceOfAging/README.md).
