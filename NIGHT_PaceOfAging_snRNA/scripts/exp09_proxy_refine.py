import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel, Bagged

ids = ek.dev_donors(); cov = ek.covariates(ids)
def lc(ct, cpm):
    L = ek.pseudobulk_libcorr(ids, ct, min_cpm=cpm); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]; return L
L3 = lc("ALL", 3)
X = pd.concat([cov, L3], axis=1)
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
ek.evaluate("Y01 proxy[clin13]+direct libcorr3 weighted sqrt", X, lambda: ProxyModel(aux=ek.CLIN13, weight="sqrt"))
ek.evaluate("Y02 proxy[clin13+blood5]+direct libcorr3", X, lambda: ProxyModel(aux=ek.CLIN13 + BLOOD))
ek.evaluate("Y03 proxy[clin13]+direct libcorr1", pd.concat([cov, lc("ALL", 1)], axis=1), lambda: ProxyModel(aux=ek.CLIN13))
X4 = pd.concat([cov, L3, lc("CD4 Active", 3)], axis=1)
ek.evaluate("Y04 proxy[clin13]+direct libcorr3 ALL+CD4A", X4, lambda: ProxyModel(aux=ek.CLIN13))
ek.evaluate("Y05 bagged5 proxy[clin13]+direct libcorr3", X, lambda: Bagged(make=lambda s: ProxyModel(aux=ek.CLIN13, seed=s), n=5))
PACE_REL = ["bmi26", "hdl26", "CurrentSmk26", "PackYrsBy26", "glyhaem26", "wbc26", "LnCRP26", "VOmax26", "trigly26"]
ek.evaluate("Y06 proxy[9 pace-related]+direct libcorr3", X, lambda: ProxyModel(aux=PACE_REL))
