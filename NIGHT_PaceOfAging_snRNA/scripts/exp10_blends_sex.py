import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel, BySex

ids = ek.dev_donors(); cov = ek.covariates(ids)
L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
X = pd.concat([cov, L], axis=1)
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
# done: ek.evaluate("S01 by-sex proxy[clin13+blood5]+direct libcorr3", X, lambda: BySex(make=lambda: ProxyModel(aux=ek.CLIN13 + BLOOD)))
# done: ek.evaluate("S02 libcorr3 +cov | pls10", X, pls(10))
# done: ek.evaluate("S03 libcorr3 +cov | pls15", X, pls(15))
# done: ek.evaluate("S04 comp celltype+leiden10 +cov | ridge", pd.concat([cov, ek.composition(ids, "celltype"), ek.composition(ids, "leiden10")], axis=1), ridge())
# blends of different snRNA-only model families (fixed equal weights)
ek.evaluate_blend("BL1 blend proxyY02 + pls10(pbALL3) + ridge-weighted(pbALL3)",
                  ["Y02 proxy[clin13+blood5]+direct libcorr3", "R09 pbALL3 +cov | pls10", "W01 pbALL3 +cov | ridge weighted sqrt"])
ek.evaluate_blend("BL2 blend proxyY02 + xgb300",
                  ["Y02 proxy[clin13+blood5]+direct libcorr3", "R07 pbALL3 +cov | screen300 xgboost d2"])
ek.evaluate_blend("BL3 blend proxyY02 + comp + pls10 + xgb300",
                  ["Y02 proxy[clin13+blood5]+direct libcorr3", "C01 comp celltype+cov | ridge", "R09 pbALL3 +cov | pls10", "R07 pbALL3 +cov | screen300 xgboost d2"])
ek.evaluate_blend("BL4 blend proxy Y02+Y05+Y06+L03", ["Y02 proxy[clin13+blood5]+direct libcorr3", "Y05 bagged5 proxy[clin13]+direct libcorr3", "Y06 proxy[9 pace-related]+direct libcorr3", "L03 proxy[clin13]+direct libcorr3 | snRNA-only"])
