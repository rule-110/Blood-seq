import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel

ids = ek.dev_donors(); cov = ek.covariates(ids)
X = pd.concat([cov, ek.pseudobulk(ids, ["ALL"], min_cpm=3)], axis=1)
SLOPES = ["bmi_iz_snL", "map_iz_snL", "vox_izr_snL", "chol_iz_snL", "tri_iz_snL", "hdl_izr_snL", "urea_iz_snL", "wbc_iz_snL"]
POA_NO = [c for c in ek.load().columns if c.startswith("POA45_no")]
AGE45 = ["suPAR45", "bmip45", "HDL45", "trigly45", "choles45", "urea45", "VOmax45", "MeanArtPr45", "Ln_CRP45",
         "CurrentSmk45", "PackYrLifTm45", "CD8pCD28nCD45RAn_45"]
ek.evaluate("F01 proxy[slopes8]+direct pbALL3 | snRNA-only", X, lambda: ProxyModel(aux=SLOPES))
ek.evaluate("F02 proxy[POA_no19]+direct pbALL3 | snRNA-only", X, lambda: ProxyModel(aux=POA_NO))
ek.evaluate("F03 proxy[age45 x12]+direct pbALL3 | snRNA-only", X, lambda: ProxyModel(aux=AGE45))
ek.evaluate("F04 proxy[clin13+slopes8]+direct pbALL3 | snRNA-only", X, lambda: ProxyModel(aux=ek.CLIN13 + SLOPES))
ek.evaluate("F05 proxy[clin13+slopes8+age45]+direct pbALL3 | snRNA-only", X, lambda: ProxyModel(aux=ek.CLIN13 + SLOPES + AGE45))
ek.evaluate("F06 proxy[clin13+slopes8+age45+POA_no]+direct | snRNA-only", X, lambda: ProxyModel(aux=ek.CLIN13 + SLOPES + AGE45 + POA_NO))
