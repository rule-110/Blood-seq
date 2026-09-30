import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import evalkit as ek
from common_models import ridge
from meta_models import ProxyModel

lb = pd.read_csv(ek.RES + "LOCKBOX_results.csv")
cv = pd.read_csv(ek.RES + "results.csv").drop_duplicates("name", keep="last").set_index("name")["auc_mean"]
cvmap = {"snRNA-only: proxy[clin13+blood5]+direct, libcorr (Y02)": "Y02 proxy[clin13+blood5]+direct libcorr3",
         "snRNA-only: ridge weighted sqrt, pseudobulk (W01)": "W01 pbALL3 +cov | ridge weighted sqrt",
         "covariates only": "B00 covariates | ridge", "BMI only": "M01 clin[bmi]+cov | ridge",
         "BMI + snRNA proxy (M02)": "M02 clin[bmi] + snRNA proxy[clin13]+direct libcorr3",
         "clinical13": "B01 clin13+cov | ridge", "clinical13 + snRNA direct (K02)": "K02 clin13 + direct pbALL3 (stacked)"}
labels = {"covariates only": "covariates only (sex, QC)", "snRNA-only: ridge weighted sqrt, pseudobulk (W01)": "snRNA-seq: plain ridge",
          "snRNA-only: proxy[clin13+blood5]+direct, libcorr (Y02)": "snRNA-seq: proxy model", "BMI only": "BMI",
          "BMI + snRNA proxy (M02)": "BMI + snRNA-seq", "clinical13": "13 clinical traits",
          "clinical13 + snRNA direct (K02)": "13 clinical + snRNA-seq"}
order = list(labels)
lb = lb.set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(8, 5.0))
yy = np.arange(len(order))[::-1]
ax.hlines(yy, lb["ci_low"], lb["ci_high"], color="#9db4cf", linewidth=2, zorder=1)
ax.scatter(lb["lockbox_auc"], yy, s=70, color="#1f3b63", zorder=3, label="lock-box AUC (160 held-out donors, 95% bootstrap CI)")
ax.scatter([cv[cvmap[m]] for m in order], yy, s=40, marker="D", facecolor="white", edgecolor="#6f6f6f", zorder=2,
           label="cross-validated AUC (626 dev donors)")
for m, v in zip(yy, lb["lockbox_auc"]):
    ax.text(0.905, m, f"{v:.3f}", va="center", fontsize=9, color="#333333")
ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1)
ax.set_yticks(yy); ax.set_yticklabels([labels[m] for m in order], fontsize=9)
ax.set_xlim(0.4, 0.95); ax.set_xlabel("AUC, High (top 20%) vs Low (bottom 60%) PaceOfAgingP45")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
ax.set_title("Predicting pace of aging (26→45) from age-26 blood snRNA-seq and clinical traits", fontsize=10)
fig.tight_layout(); fig.savefig(ek.BASE + "figures/final_auc_summary.png", dpi=200); fig.savefig(ek.BASE + "figures/final_auc_summary.pdf")

import sys as _s
if "--fig-only" in _s.argv: raise SystemExit
# final models refit on ALL 786 donors (dev + lock-box) for future use
allid = ek.load().index
cov = ek.covariates(allid)
L = ek.pseudobulk_libcorr(allid, "ALL", min_cpm=3, ref_ids=allid); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
y = ek.load().reindex(allid)[ek.TARGET].to_numpy()
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
m1 = ProxyModel(aux=ek.CLIN13 + BLOOD).fit(pd.concat([cov, L], axis=1), y)
m2 = ProxyModel(aux=ek.CLIN13, extra_prefix="clin|").fit(pd.concat([cov, L, ek.clinical(allid, ["bmi26"])], axis=1), y)
with open(ek.RES + "final_models_all786.pkl", "wb") as fh:
    pickle.dump({"snRNA_only_proxy": m1, "bmi_plus_snRNA_proxy": m2, "feature_columns_snRNA": list(pd.concat([cov, L], axis=1).columns),
                 "note": "features = ek.covariates + ek.pseudobulk_libcorr(ids,'ALL',min_cpm=3) (pbL| renamed pb|); +clin|bmi26 for model 2"}, fh)
print("saved")
