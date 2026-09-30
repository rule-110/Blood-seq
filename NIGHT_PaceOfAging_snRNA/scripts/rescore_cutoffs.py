"""Re-score saved predictions (no refitting) with other case/control cutoffs. Usage: rescore_cutoffs.py QLOW QHIGH"""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import evalkit as ek
QL, QH = float(sys.argv[1]), float(sys.argv[2])
CV = {"covariates only": "B00 covariates | ridge",
      "snRNA-seq only (proxy model, Y02)": "Y02 proxy[clin13+blood5]+direct libcorr3",
      "BMI only": "M01 clin[bmi]+cov | ridge",
      "BMI + snRNA-seq (M02)": "M02 clin[bmi] + snRNA proxy[clin13]+direct libcorr3",
      "13 clinical": "B01 clin13+cov | ridge",
      "13 clinical + snRNA-seq (K02)": "K02 clin13 + direct pbALL3 (stacked)"}
LB = {"covariates only": "covariates only", "snRNA-seq only (proxy model, Y02)": "snRNA-only: proxy[clin13+blood5]+direct, libcorr (Y02)",
      "BMI only": "BMI only", "BMI + snRNA-seq (M02)": "BMI + snRNA proxy (M02)", "13 clinical": "clinical13",
      "13 clinical + snRNA-seq (K02)": "clinical13 + snRNA direct (K02)"}
ids = ek.dev_donors(); y = ek.load().reindex(ids)[ek.TARGET].to_numpy(float)
sp = ek.splits(len(ids))
def fold_auc(p_all):
    out = []
    for k, (tr, te) in enumerate(sp):
        lo, hi = np.quantile(y[tr], QL), np.quantile(y[tr], QH)
        e = (y[te] <= lo) | (y[te] >= hi); lab = (y[te][e] >= hi).astype(int)
        p = p_all[k // ek.N_FOLDS, te][e]
        out.append(roc_auc_score(lab, p) if 0 < lab.sum() < len(lab) else np.nan)
    return np.array(out)
cvres = {n: fold_auc(np.load(ek.RES + f"oof/{ek.safe(f)}.npz", allow_pickle=True)["oof"]) for n, f in CV.items()}
lp = pd.read_csv(ek.RES + "LOCKBOX_predictions.csv", index_col=0)
ydev = ek.load().reindex(ids)[ek.TARGET]
lo, hi = ydev.quantile(QL), ydev.quantile(QH)
yl = lp["pace"].to_numpy(); e = (yl <= lo) | (yl >= hi); lab = (yl[e] >= hi).astype(int)
rng = np.random.default_rng(0); boots = [rng.integers(0, e.sum(), e.sum()) for _ in range(2000)]
print(f"cutoffs: Low <= {QL:.0%} quantile, High >= {QH:.0%} quantile")
print(f"lock-box: {lab.sum()} High vs {(lab == 0).sum()} Low | CV test folds: ~{int(round(len(ids)/5*(1-QH)))} High per fold\n")
rows = []
for n in CV:
    p = lp[LB[n]].to_numpy()[e]
    bs = [roc_auc_score(lab[i], p[i]) for i in boots if 0 < lab[i].sum() < len(i)]
    rows.append({"model": n, "CV_auc": np.nanmean(cvres[n]), "CV_sd": np.nanstd(cvres[n]),
                 "lockbox_auc": roc_auc_score(lab, p), "lb_ci_low": np.percentile(bs, 2.5), "lb_ci_high": np.percentile(bs, 97.5)})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
def paired(a, b):
    d = (cvres[a] - cvres[b]); d = d[~np.isnan(d)]
    t = d.mean() / np.sqrt(np.var(d, ddof=1) * (1 / len(d) + 0.25))
    pa, pb = lp[LB[a]].to_numpy()[e], lp[LB[b]].to_numpy()[e]
    bd = np.array([roc_auc_score(lab[i], pa[i]) - roc_auc_score(lab[i], pb[i]) for i in boots if 0 < lab[i].sum() < len(i)])
    return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1), roc_auc_score(lab, pa) - roc_auc_score(lab, pb), np.mean(bd <= 0)
print()
for a, b in [("snRNA-seq only (proxy model, Y02)", "covariates only"), ("BMI + snRNA-seq (M02)", "BMI only"), ("13 clinical + snRNA-seq (K02)", "13 clinical")]:
    dc, pc, dl, pl = paired(a, b)
    print(f"{a} vs {b}: CV dAUC {dc:+.3f} (p={pc:.3g}) | lock-box dAUC {dl:+.3f} (one-sided boot p={pl:.3g})")
res.to_csv(ek.RES + f"rescore_low{int(QL*100)}_high{int(QH*100)}.csv", index=False)
