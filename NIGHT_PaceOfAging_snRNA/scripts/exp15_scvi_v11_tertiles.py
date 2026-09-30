"""v11 pipeline (scVI-denoised expression per cell type -> stability-selection XGBoost) trained ONCE per CV fold on
all dev donors, then scored within HDL / CRP tertiles, next to the smoking-only (B07) and snRNA proxy (Y02) models.

Same folds as exp01-14. High/Low = train-fold 60% / 90% quantiles (v11 definition; exp01-14 used 60/80). The classifier is fit on the training
fold's extreme donors, as in v11. Genes = top 1000 by mean raw count in that cell type (v11 rule; label-free, so it is
computed once on all cells, whereas v11 used train cells only).
Stability selection and final-model hyper-parameters are copied from run_discovery_pdf_report (v2) in notebook (11).
"""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd, scanpy as sc
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import evalkit as ek
ek.Q_LOW, ek.Q_HIGH = 0.6, 0.9      # v11 definition: High = top 10%, Low = bottom 60%

N_ITER, STAB_THRESH, N_TOP = 50, 60, 1000
CELL_TYPES = ["CD4 Active", "Naive T cells", "CD8 Active", "B Naive", "B Memory", "NK cells",
              "Classical Monocytes", "Non-classical Monocytes", "MAIT cells"]


class V11XGB(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y):
        X = X.astype(np.float32); y = np.asarray(y).astype(int)
        counts = pd.Series(0, index=X.columns)
        for i in range(N_ITER):
            s = self.seed + i
            Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
            clf = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.6,
                                device="cuda", tree_method="hist", eval_metric="logloss", random_state=s)
            clf.fit(Xtr, ytr)
            counts[X.columns[clf.feature_importances_ > 0]] += 1
        pct = counts / N_ITER * 100
        genes = pct[pct >= STAB_THRESH].index.tolist()
        if len(genes) < 2:
            genes = pct.sort_values(ascending=False).head(10).index.tolist()
        self.genes_ = genes
        self.clf_ = XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                  device="cuda", tree_method="hist", eval_metric="logloss", random_state=42)
        self.clf_.fit(X[genes], y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X[self.genes_].astype(np.float32))


ids = ek.dev_donors()
Z = np.load(ek.DATA + "scvi_denoised_ct.npz", allow_pickle=True)
genes_all = Z["genes"]
ad = sc.read_h5ad("/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad")   # raw counts in .X (v11 gene rule)
cts = ad.obs["CellType"].astype(str).to_numpy()

names = {}
for c in CELL_TYPES:
    key = c.replace(" ", "_")
    means = np.asarray(ad.X[cts == c].mean(axis=0)).ravel()
    top = np.argsort(means)[::-1][:N_TOP]
    df = pd.DataFrame(Z[f"{key}__mean"][:, top], index=Z[f"{key}__donors"], columns=[f"scvi|{g}" for g in genes_all[top]])
    X = df.reindex(ids)
    print(f"{c}: {X.notna().all(axis=1).sum()}/{len(ids)} dev donors have this cell type", flush=True)
    name = f"V{len(names)+1:02d} v11 scVI-XGB q60-90 {c}"
    ek.evaluate(name, X, lambda: V11XGB(), task="clf", notes="exp15 v11 pipeline, 1 model on all dev donors")
    names[c] = name
del ad

# ---------------- subgroup scoring ----------------
d = ek.load().reindex(ids); y = d[ek.TARGET].to_numpy()
lo, hi = np.quantile(y, ek.Q_LOW), np.quantile(y, ek.Q_HIGH); ext = (y <= lo) | (y >= hi); lab = (y >= hi).astype(int)
oof = {c: np.load(ek.RES + f"oof/{ek.safe(n)}.npz")["oof"] for c, n in names.items()}
# ensemble over cell types: mean of within-repeat ranks (donors missing a cell type just skip it)
rk = np.stack([pd.DataFrame(o.T).rank(pct=True).to_numpy().T for o in oof.values()])
oof["scVI ensemble (9 CT)"] = np.nanmean(rk, axis=0)
oof["snRNA proxy (Y02)"] = np.load(ek.RES + f"oof/{ek.safe('Y02 proxy[clin13+blood5]+direct libcorr3')}.npz")["oof"]
oof["smoking only (B07)"] = np.load(ek.RES + f"oof/{ek.safe('B07 smoking only+cov | ridge')}.npz")["oof"]


def A(p, m):
    m = m & ~np.isnan(p).any(axis=0)
    return np.mean([roc_auc_score(lab[m], p[r][m]) for r in range(p.shape[0])]) if 0 < lab[m].sum() < m.sum() else np.nan


rng = np.random.default_rng(0)
rows = []
for trait in ["hdl26", "LnCRP26"]:
    t = pd.qcut(d[trait], 3, labels=["Low", "Medium", "High"]).to_numpy()
    for model, p in oof.items():
        r = {"trait": trait, "model": model, "all": A(p, ext)}
        for lv in ["Low", "Medium", "High"]:
            r[lv] = A(p, ext & (t == lv))
        L = np.where(ext & (t == "Low") & ~np.isnan(p).any(0))[0]; H = np.where(ext & (t == "High") & ~np.isnan(p).any(0))[0]
        bs = []
        for _ in range(1000):
            l, h = rng.choice(L, len(L)), rng.choice(H, len(H))
            if 0 < lab[l].sum() < len(l) and 0 < lab[h].sum() < len(h):
                bs.append(np.mean([roc_auc_score(lab[l], p[k][l]) - roc_auc_score(lab[h], p[k][h]) for k in range(p.shape[0])]))
        r["Low-High"] = r["Low"] - r["High"]
        r["CI95"] = f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
        rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(ek.RES + "exp15_tertile_auc_q60-90.csv", index=False)
with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
    for trait, g in tab.groupby("trait", sort=False):
        print(f"\n===== AUC within {trait} tertiles (one model on all dev donors, OOF, global cutoffs 0.6/0.9) =====")
        print(g.drop(columns="trait").set_index("model").to_string())
