import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel, Weighted

ids = ek.dev_donors(); cov = ek.covariates(ids)
pb3 = ek.pseudobulk(ids, ["ALL"], min_cpm=3)
Xp = pd.concat([cov, pb3], axis=1)
for mode in ["sqrt", "log", "min100", "cap"]:
    ek.evaluate(f"W01 pbALL3 +cov | ridge weighted {mode}", Xp, lambda mode=mode: Weighted(make=ridge(), mode=mode))
for cpm in (1, 3, 10):
    L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=cpm).add_prefix("")
    L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
    XL = pd.concat([cov, L], axis=1)
    ek.evaluate(f"L01 pbALL libcorr cpm>={cpm} +cov | ridge", XL, ridge())
    if cpm == 3:
        ek.evaluate("L02 pbALL libcorr cpm>=3 +cov | ridge weighted sqrt", XL, lambda: Weighted(make=ridge(), mode="sqrt"))
        ek.evaluate("L03 proxy[clin13]+direct libcorr3 | snRNA-only", XL, lambda: ProxyModel(aux=ek.CLIN13))
L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3, weight="equal")
L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
ek.evaluate("L04 pbALL libcorr cpm>=3 equal-lib-weight +cov | ridge", pd.concat([cov, L], axis=1), ridge())
