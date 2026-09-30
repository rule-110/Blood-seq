import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel

ids = ek.dev_donors(); cov = ek.covariates(ids)
L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
SUBSETS = {"bmi": ["bmi26"], "bmi+smk": ["bmi26", "CurrentSmk26", "PackYrsBy26"],
           "bmi+hdl": ["bmi26", "hdl26"], "bmi+smk+hdl": ["bmi26", "CurrentSmk26", "PackYrsBy26", "hdl26"],
           "noninvasive(bmi,smk,MAP)": ["bmi26", "CurrentSmk26", "PackYrsBy26", "MeanArtPr26"]}
for nm, cols in SUBSETS.items():
    c = ek.clinical(ids, cols)
    ek.evaluate(f"M01 clin[{nm}]+cov | ridge", pd.concat([cov, c], axis=1), ridge())
    ek.evaluate(f"M02 clin[{nm}] + snRNA proxy[clin13]+direct libcorr3", pd.concat([cov, L, c], axis=1),
                lambda: ProxyModel(aux=ek.CLIN13, extra_prefix="clin|"))
