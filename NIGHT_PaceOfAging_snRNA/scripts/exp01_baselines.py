import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd
import evalkit as ek
from common_models import *

ek.make_lockbox()
ids = ek.dev_donors()
print(f"dev donors: {len(ids)}, lockbox: {len(ek.lock_donors())}", flush=True)
cov = ek.covariates(ids)
clin = ek.clinical(ids)
clinx = ek.clinical(ids, ek.CLIN13 + ek.CLIN_EXTRA)
clind = ek.clinical(ids, ek.CLIN13 + ek.CLIN_EXTRA + ek.DNAM_CELLS)
ek.evaluate("B00 covariates | ridge", cov, ridge())
ek.evaluate("B01 clin13+cov | ridge", pd.concat([cov, clin], axis=1), ridge())
ek.evaluate("B02 clin13+cov | enet", pd.concat([cov, clin], axis=1), enet())
ek.evaluate("B03 clin13+cov | logitL2 clf", pd.concat([cov, clin], axis=1), logit_l2(), task="clf")
ek.evaluate("B04 clin13+extra+cov | ridge", pd.concat([cov, clinx], axis=1), ridge())
ek.evaluate("B05 clin13+extra+DNAmcells+cov | ridge", pd.concat([cov, clind], axis=1), ridge())
ek.evaluate("B06 clin13 no smoking+cov | ridge", pd.concat([cov, ek.clinical(ids, [c for c in ek.CLIN13 if c not in ek.SMOKING])], axis=1), ridge())
ek.evaluate("B07 smoking only+cov | ridge", pd.concat([cov, ek.clinical(ids, ek.SMOKING)], axis=1), ridge())
for w in ["celltype", "leiden05", "leiden10"]:
    ek.evaluate(f"C01 comp {w}+cov | ridge", pd.concat([cov, ek.composition(ids, w)], axis=1), ridge())
pb_all = ek.pseudobulk(ids, ["ALL"], min_cpm=10)
print("pb ALL", pb_all.shape, flush=True)
ek.evaluate("P01 pbALL logcpm cpm>=10 +cov | ridge", pd.concat([cov, pb_all], axis=1), ridge())
ek.evaluate("P02 pbALL logcpm cpm>=10 +cov | pca50 ridge", pd.concat([cov, pb_all], axis=1), pca_ridge(50))
ek.evaluate("P03 pbALL logcpm cpm>=10 +cov | pls3", pd.concat([cov, pb_all], axis=1), pls(3))
ek.evaluate("P04 pbALL logcpm cpm>=10 +cov | screen500 ridge", pd.concat([cov, pb_all], axis=1), screen_ridge(500))
