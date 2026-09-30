import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
from meta_models import ProxyModel, ViewEnsemble

ids = ek.dev_donors(); cov = ek.covariates(ids)
d = ek.load().reindex(ids)
pb3 = ek.pseudobulk(ids, ["ALL"], min_cpm=3)
Xall = pd.concat([cov, pb3], axis=1)

# how well does snRNA-seq itself predict each age-26 trait? (sanity + guides the proxy set)
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats
rows = []
for t in [] if "--skip" in sys.argv else ek.CLIN13 + ek.CLIN_EXTRA + ek.DNAM_CELLS:
    y = pd.to_numeric(d[t], errors="coerce"); ok = y.notna().to_numpy()
    p = cross_val_predict(ridge()(), Xall[ok], y[ok], cv=KFold(5, shuffle=True, random_state=0))
    rows.append({"trait": t, "n": ok.sum(), "spearman_snRNA": stats.spearmanr(y[ok], p)[0],
                 "spearman_trait_vs_pace": stats.spearmanr(y[ok], d[ek.TARGET][ok])[0]})
tr = pd.DataFrame(rows, columns=["trait","n","spearman_snRNA","spearman_trait_vs_pace"]).sort_values("spearman_snRNA", ascending=False)
if len(tr): tr.to_csv(ek.RES + "snRNA_predicts_traits.csv", index=False)
print(tr.round(3).to_string(index=False), flush=True)

AUX_ALL = ek.CLIN13 + ek.CLIN_EXTRA + ek.DNAM_CELLS
AUX_13 = ek.CLIN13
if "--skip" not in sys.argv: ek.evaluate("X01 proxy[clin13]+direct pbALL3 | snRNA-only", Xall, lambda: ProxyModel(aux=AUX_13))
if "--skip" not in sys.argv: ek.evaluate("X02 proxy[all31]+direct pbALL3 | snRNA-only", Xall, lambda: ProxyModel(aux=AUX_ALL))
if "--skip" not in sys.argv: ek.evaluate("X03 proxy[clin13] no direct pbALL3 | snRNA-only", Xall, lambda: ProxyModel(aux=AUX_13, keep_direct=False))
ek.evaluate("X04 direct only (stacked ridge) pbALL3 | snRNA-only", Xall, lambda: ProxyModel(aux=()))

# per-cell-type view ensembles
CTS = ["ALL", "CD4 Active", "Naive T cells", "B Naive", "B Memory", "Classical Monocytes", "CD8 Active", "NK cells"]
pbv = ek.pseudobulk(ids, CTS, min_cells=10, min_frac_donors=0.3, min_cpm=3)
comp = ek.composition(ids, "celltype")
Xv = pd.concat([cov, pbv, comp], axis=1)
views = [f"pb|{c}|" for c in CTS] + ["clr_celltype|"]
ek.evaluate("V01 views(8CT+comp) mean | snRNA-only", Xv, lambda: ViewEnsemble(views=views, combine="mean"))
ek.evaluate("V02 views(8CT+comp) stack | snRNA-only", Xv, lambda: ViewEnsemble(views=views, combine="stack"))
views5 = [f"pb|{c}|" for c in ["ALL", "CD4 Active", "Naive T cells", "B Naive", "B Memory"]]
ek.evaluate("V03 views(ALL,CD4A,NaiveT,BN,BM) mean | snRNA-only", Xv, lambda: ViewEnsemble(views=views5, combine="mean"))
