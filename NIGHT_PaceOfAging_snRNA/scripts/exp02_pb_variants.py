import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *

ids = ek.dev_donors(); cov = ek.covariates(ids)
CTS = ["CD4 Active", "Naive T cells", "CD8 Active", "B Naive", "B Memory", "NK cells", "Classical Monocytes"]
# each cell type alone (donors lacking it are imputed)
for ct in CTS:
    for mc in (10,):
        X = ek.pseudobulk(ids, [ct], min_cells=mc, min_frac_donors=0.3, min_cpm=10)
        if X is None or X.shape[1] == 0: continue
        cov_ct = X.notna().any(axis=1).mean()
        ek.evaluate(f"P10 pb {ct} minc{mc} logcpm +cov | ridge", pd.concat([cov, X], axis=1), ridge(), notes=f"coverage {cov_ct:.2f}")
# gene-expression thresholds on ALL
for cpm in (1, 3, 30, 100):
    X = ek.pseudobulk(ids, ["ALL"], min_cpm=cpm)
    ek.evaluate(f"P11 pbALL logcpm cpm>={cpm} +cov | ridge", pd.concat([cov, X], axis=1), ridge())
# detection-rate representation
X = ek.pseudobulk(ids, ["ALL"], min_cpm=10, kind="det")
ek.evaluate("P12 pbALL det cpm>=10 +cov | ridge", pd.concat([cov, X], axis=1), ridge())
# concatenated cell types
X = ek.pseudobulk(ids, ["ALL", "CD4 Active", "Naive T cells", "CD8 Active", "B Naive"], min_cells=20, min_frac_donors=0.5, min_cpm=10)
ek.evaluate("P13 pb ALL+CD4A+NaiveT+CD8A+BNaive minc20 +cov | ridge", pd.concat([cov, X], axis=1), ridge())
X = ek.pseudobulk(ids, ["ALL"] + CTS, min_cells=10, min_frac_donors=0.3, min_cpm=10)
ek.evaluate("P14 pb ALL+7CT minc10 +cov | ridge", pd.concat([cov, X], axis=1), ridge())
