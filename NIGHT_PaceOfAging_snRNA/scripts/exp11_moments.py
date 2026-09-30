import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel

ids = ek.dev_donors(); cov = ek.covariates(ids)
for ct in ["ALL", "CD4 Active", "Naive T cells", "B Naive"]:
    for kind in ["mean", "var"]:
        M = ek.lognorm_moments(ids, ct, kind)
        ek.evaluate(f"N01 ln{kind} {ct} cpm3 +cov | ridge", pd.concat([cov, M], axis=1), ridge())
M = ek.lognorm_moments(ids, "ALL", "mean"); V = ek.lognorm_moments(ids, "ALL", "var")
ek.evaluate("N02 lnmean+lnvar ALL +cov | ridge", pd.concat([cov, M, V], axis=1), ridge())
Mp = M.copy(); Mp.columns = [c.replace("lnmean|", "pb|") for c in M.columns]
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
ek.evaluate("N03 proxy[clin13+blood5]+direct lnmean ALL", pd.concat([cov, Mp], axis=1), lambda: ProxyModel(aux=ek.CLIN13 + BLOOD))
Vp = V.copy(); Vp.columns = [c.replace("lnvar|", "pb|") for c in V.columns]
ek.evaluate("N04 proxy[clin13+blood5]+direct lnmean+lnvar ALL", pd.concat([cov, Mp, Vp.add_suffix("_v")], axis=1), lambda: ProxyModel(aux=ek.CLIN13 + BLOOD))
