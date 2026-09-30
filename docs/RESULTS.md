# Results log (PaceOfAgingP45)

A condensed record of what was tried and what came out of it, in chronological order.
"AUC" means High (fast agers) vs Low (slow agers) on PaceOfAgingP45. The quantile cutoffs
are given for each result.

## 1. Per-cell-type scVI pseudobulk + XGBoost (v11 notebook)

- Cell-type × sex multi-split: test AUC 0.43–0.64 (SD 0.04–0.14), train AUC 1.0 everywhere.
  The signal is weak and the models overfit heavily.
- **Tertile sweeps** (label donors inside HDL / CRP tertiles). The best hits look strong but are noisy:
  - CD8 Active / hdl26 / Low fell from 0.85 (3 splits) to 0.69 ± 0.09 (10 splits).
    Its test set has only ~7 High vs 46 Low donors.
  - Tertile AUC tracks the within-tertile High−Low pace gap (ρ = 0.53), so per-tertile labels
    partly confound the comparison between tertiles (`label_cutoffs="global"` fixes this).
  - crp26ns (20 splits): Low CRP is best in 8 of 9 cell types (~0.62 vs ~0.49 / 0.52).
    `POA45_nocrp` shows the same gradient.
  - The same split's AUC can move by up to 0.4 between runs when only the scVI denoising draw and
    the XGBoost device change. Per-split AUCs are dominated by model randomness.
  - When the rows are ranked by test AUC, the top rows carry a winner's-curse boost of about +0.04.
- **HDL / CRP gradient checks** on out-of-fold predictions: the Low-HDL advantage follows
  smoking (a smoking-only model shows the same gradient). The CRP gradient is borderline and
  post hoc (proxy model 0.76 in Low vs 0.64 in High; CI −0.01 to 0.25).

## 2. Clinical vs snRNA-seq, one row per donor (v11 cells 83–93)

Elastic net / ridge on continuous pace, 5×5 donor-level CV.

| features | R² | Spearman | AUC |
|---|---|---|---|
| age-26 clinical | 0.25 | 0.48 | 0.80 |
| scVI latent + composition | 0.00 | 0.14 | 0.59 |
| pseudobulk genes (6 cell groups, 6445 genes) | 0.036 | 0.23 | 0.65 |

Neither snRNA block adds to clinical (ΔR² ≈ −0.005 to −0.007). A clinical → snRNA cascade
on the 40 % least-confident donors lowers AUC (0.79 vs 0.81).

## 3. Systematic model search with lock-box (`NIGHT_PaceOfAging_snRNA/`)

About 90 pipelines were compared on 626 dev donors (5×3 CV). The pre-declared final models
were then scored **once** on 160 lock-box donors. The table is in the main README.
- The best snRNA-only model is a **proxy model**: ridge predicts 18 age-26 traits plus pace from
  library-corrected pseudobulk, then a second-level model combines them. Lock-box AUC 0.70, ρ 0.29 (p < 0.001).
- BMI + snRNA vs BMI: +0.075 AUC (p = 0.06). Clinical + snRNA vs clinical: −0.007.
- The signal is mostly smoking: it disappears for `POA45_nocal` and on smoking-residualized pace.

## 4. Other feature sources (all negative)

| approach | result |
|---|---|
| v12 proxy features + XGB (10 splits) | 0.67 ± 0.03 (ridge second level: 0.70) |
| CNA (co-varying neighbourhood abundance) | only smoking significant |
| sc-ImmuAging clocks (CD4T/CD8T/MONO/NK/B) | biologically sensible (vs DNAm naive CD8 ρ ≈ −0.35) but AUC 0.52–0.55 for pace |
| MIL on scVI latent (q 0.6/0.9) | mean-pool 0.67, ABMIL 0.66, MixMIL 0.63, scMILD 0.60, ScRAT 0.59, CloudPred 0.53, vs proxy 0.75 |
| single v11 scVI-XGB trained once on all dev donors | 0.51–0.64 per cell type, ensemble 0.59 |

## 5. What snRNA-seq at 26 *can* predict (trait scan, `exp16_trait_scan.py`)

Sex-adjusted Spearman ρ, all-cells pseudobulk ridge. Adjusting for sex matters: VO2max's
ρ of 0.70 is pure sex.
- Best: DNAm NK 0.52, CD8 naive 0.51, B 0.49, smoking 0.46 (AUC 0.83), pack-years 0.45,
  eosinophils 0.43, WBC 0.40, HbA1c 0.39, CRP 0.38.
- Weak: BMI 0.15, cholesterol 0.11, HDL 0.11, blood pressure ≈ 0.
- Pace 0.27. All `POA45_no*` variants give 0.24–0.29, except `nocal` at 0.19.
- Change after 26 (adjusted for sex and the trait's own age-26 value): all ≤ 0.21. Slopes ≤ 0.11.

## Open items

- Donor-level random-tertile permutation test (`PERMTEST_tertile_scvi/`): the cache is built,
  but the run was interrupted by a machine crash. Relaunch with `python permtest_tertile.py run 60`.
- Full no-scVI tertile sweep (`notebooks/biomarker_noscvi_pseudobulk_tertile.ipynb`) has not been run yet.
- The v12 notebook edits for genes / hvg2000 mode (cells 95/98/99) were lost to an overwrite and need re-applying.
