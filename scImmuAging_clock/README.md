# sc-ImmuAging clock coefficients

`clock_coefs_symbols.csv` holds the LASSO (cv.glmnet, lambda.min) coefficients of the sc-ImmuAging
CD4T / CD8T / MONO / NK / B clocks (Li et al., *Nature Aging* 2025), with Ensembl GRCh38 IDs
mapped to gene symbols through the 10x hg19 `features.tsv`.

Source: https://github.com/CiiM-Bioinformatics-group/scImmuAging, licensed under Apache-2.0.
This is a derived file. Cite the original paper when you use it.

Used by `NIGHT_PaceOfAging_snRNA/scripts/prep5_scimmuaging.py` and `exp14_scimmuaging.py`. It needs the
full-transcriptome h5ad (82–98 % of |coef| mass covered); the HVG-2000 file covers only 5–30 %.
