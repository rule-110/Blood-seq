"""ONE-TIME evaluation of pre-declared final models on the 160 lock-box donors.
Models are trained on all 626 dev donors. Library correction uses dev donors only.
High/Low cutoffs (0.6 / 0.8 quantiles) come from the dev donors' pace."""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import evalkit as ek
from common_models import ridge
from meta_models import ProxyModel, Weighted

dev, lock = ek.dev_donors(), ek.lock_donors()
allid = dev.append(lock)
cov = ek.covariates(allid)
L = ek.pseudobulk_libcorr(allid, "ALL", min_cpm=3, ref_ids=dev); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
P3 = ek.pseudobulk(allid, ["ALL"], min_cpm=3)
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
y = ek.load().reindex(allid)[ek.TARGET]

MODELS = {  # name: (X, factory) -- declared before looking at the lock-box
    "snRNA-only: proxy[clin13+blood5]+direct, libcorr (Y02)": (pd.concat([cov, L], axis=1), lambda: ProxyModel(aux=ek.CLIN13 + BLOOD)),
    "snRNA-only: ridge weighted sqrt, pseudobulk (W01)": (pd.concat([cov, P3], axis=1), lambda: Weighted(make=ridge(), mode="sqrt")),
    "covariates only": (cov, ridge()),
    "BMI only": (pd.concat([cov, ek.clinical(allid, ["bmi26"])], axis=1), ridge()),
    "BMI + snRNA proxy (M02)": (pd.concat([cov, L, ek.clinical(allid, ["bmi26"])], axis=1), lambda: ProxyModel(aux=ek.CLIN13, extra_prefix="clin|")),
    "clinical13": (pd.concat([cov, ek.clinical(allid)], axis=1), ridge()),
    "clinical13 + snRNA direct (K02)": (pd.concat([cov, P3, ek.clinical(allid)], axis=1), lambda: ProxyModel(aux=(), extra_prefix="clin|")),
}
lo, hi = y[dev].quantile(ek.Q_LOW), y[dev].quantile(ek.Q_HIGH)
yl = y[lock].to_numpy()
ext = (yl <= lo) | (yl >= hi)
lab = (yl[ext] >= hi).astype(int)
print(f"lock-box: {len(lock)} donors, {lab.sum()} High vs {(lab == 0).sum()} Low (middle {(~ext).sum()} excluded from AUC)", flush=True)
preds = {}
for name, (X, make) in MODELS.items():
    m = make().fit(X.loc[dev], y[dev].to_numpy())
    preds[name] = m.predict(X.loc[lock])
rng = np.random.default_rng(0)
B = 2000
boot_idx = [rng.integers(0, ext.sum(), ext.sum()) for _ in range(B)]
rows = []
for name, p in preds.items():
    pe = p[ext]
    auc = roc_auc_score(lab, pe)
    bs = [roc_auc_score(lab[i], pe[i]) for i in boot_idx if 0 < lab[i].sum() < len(i)]
    rho = stats.spearmanr(yl, p)[0]
    rows.append({"model": name, "lockbox_auc": auc, "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                 "lockbox_spearman": rho, "spearman_p": stats.spearmanr(yl, p)[1]})
res = pd.DataFrame(rows)
pairs = [("BMI + snRNA proxy (M02)", "BMI only"), ("clinical13 + snRNA direct (K02)", "clinical13"),
         ("snRNA-only: proxy[clin13+blood5]+direct, libcorr (Y02)", "covariates only")]
prow = []
for a, b in pairs:
    pa, pb = preds[a][ext], preds[b][ext]
    d = [roc_auc_score(lab[i], pa[i]) - roc_auc_score(lab[i], pb[i]) for i in boot_idx if 0 < lab[i].sum() < len(i)]
    prow.append({"comparison": f"{a}  minus  {b}", "delta_auc": roc_auc_score(lab, pa) - roc_auc_score(lab, pb),
                 "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5), "p_boot_one_sided": np.mean(np.array(d) <= 0)})
pres = pd.DataFrame(prow)
res.to_csv(ek.RES + "LOCKBOX_results.csv", index=False)
pres.to_csv(ek.RES + "LOCKBOX_comparisons.csv", index=False)
pd.DataFrame(preds, index=lock).assign(pace=yl).to_csv(ek.RES + "LOCKBOX_predictions.csv")
print(res.round(3).to_string(index=False))
print(pres.round(3).to_string(index=False))
