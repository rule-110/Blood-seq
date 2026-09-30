import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel, ViewEnsemble
import genesets as gs

ids = ek.dev_donors(); cov = ek.covariates(ids); clin = ek.clinical(ids)
CTS = ["ALL", "CD4 Active", "Naive T cells", "CD8 Active", "B Naive", "B Memory", "NK cells", "Classical Monocytes"]
G = ek.pseudobulk(ids, CTS, min_cells=10, min_frac_donors=0.3, min_cpm=0, genes=gs.UNION)
print("curated genes x CT:", G.shape, flush=True)
ek.evaluate("G01 curated union x 8CT +cov | ridge", pd.concat([cov, G], axis=1), ridge())
ek.evaluate("G02 curated union x ALL +cov | ridge", pd.concat([cov, G.filter(like="pb|ALL|")], axis=1), ridge())
for nm, genes in gs.ALL_SETS.items():
    Xs = G[[c for c in G.columns if c.split("|")[2] in genes]]
    ek.evaluate(f"G03 set {nm} x 8CT +cov | ridge", pd.concat([cov, Xs], axis=1), ridge())
# mean z-score per set x cell type (very low dimensional)
sc = {}
for nm, genes in gs.ALL_SETS.items():
    for ct in CTS:
        cols = [c for c in G.columns if c.startswith(f"pb|{ct}|") and c.split("|")[2] in genes]
        if cols:
            Z = G[cols]; Z = (Z - Z.mean()) / (Z.std() + 1e-9)
            sc[f"score|{ct}|{nm}"] = Z.mean(axis=1)
S = pd.DataFrame(sc)
ek.evaluate("G04 set scores (7 sets x 8CT) +cov | ridge", pd.concat([cov, S], axis=1), ridge())

# combined with measured clinical
pb3 = ek.pseudobulk(ids, ["ALL"], min_cpm=3)
Xc = pd.concat([cov, pb3, clin], axis=1)
ek.evaluate("K01 clin13 + proxy[clin13]+direct pbALL3 (stacked)", Xc, lambda: ProxyModel(aux=ek.CLIN13, extra_prefix="clin|"))
ek.evaluate("K02 clin13 + direct pbALL3 (stacked)", Xc, lambda: ProxyModel(aux=(), extra_prefix="clin|"))
ek.evaluate("K03 clin13 + curated union 8CT +cov | ridge", pd.concat([cov, clin, G], axis=1), ridge())
ek.evaluate("K04 clin13 + set scores +cov | ridge", pd.concat([cov, clin, S], axis=1), ridge())
