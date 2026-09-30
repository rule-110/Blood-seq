import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import *
ids = ek.dev_donors(); cov = ek.covariates(ids)
L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3)
X = pd.concat([cov, L], axis=1); Xc = pd.concat([cov, ek.clinical(ids)], axis=1)
d = ek.load().reindex(ids)
rows = []
for t in [ek.TARGET] + [c for c in d.columns if c.startswith("POA45_no")]:
    y = d[t]
    ok = y.notna()
    if ok.mean() < 0.95: continue
    y = y.fillna(y.median())
    a = ek.evaluate(f"T01 snRNA libcorr3 ridge -> {t}", X, ridge(), y=y, save=False, verbose=False)
    b = ek.evaluate(f"T02 clin13 ridge -> {t}", Xc, ridge(), y=y, save=False, verbose=False)
    rows.append({"target": t, "snRNA_auc": a["auc_mean"], "snRNA_rho": a["spearman_mean"], "clin_auc": b["auc_mean"], "clin_rho": b["spearman_mean"],
                 "corr_with_pace": np.corrcoef(d.loc[ok, t], d.loc[ok, ek.TARGET])[0, 1]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).sort_values("snRNA_auc", ascending=False).to_csv(ek.RES + "targets_component_removal.csv", index=False)
print(pd.DataFrame(rows).sort_values("snRNA_auc", ascending=False).round(3).to_string(index=False))
