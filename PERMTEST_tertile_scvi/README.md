# Random-tertile permutation test

Tests whether "AUC in the Low tertile minus AUC in the High tertile, averaged over 9 cell types"
(hdl26, LnCRP26) is larger than what random groups of the same sizes give. It execs the
unchanged v11 `run_discovery_pdf_report` (cell 58 of `notebooks/biomarker_v11_scvi_pseudobulk_ML.ipynb`).

```bash
python permtest_tertile.py cache      # scVI-denoised expression cache (GPU, ~3.4 GB)
python permtest_tertile.py run 60     # perm 0 = observed; resumes from results/rows.csv
python permtest_tertile.py summary
```
