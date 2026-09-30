"""Which phenotype can snRNA-seq (age 26) model best?  (2026-09-29)

For every numeric column of the Dunedin phenotype file, on the 626 dev donors (lock-box untouched),
5-fold x 3 repeats, all-cells pseudobulk (min CPM 3) + covariates, ridge:
  rho_cov      covariates only (sex, log n cells, log UMI, % mt)
  rho_snRNA    covariates + pseudobulk
  rho_sexadj   pseudobulk+cov predicting the trait residualized on sex (OLS fit in the training fold)
  rho_b26adj   later traits only: residualized on sex + the trait's own age-26 value (= predicts change)
  rho_trait26  later traits only: Spearman between the trait and its own age-26 value (tracking)
Output: results/trait_scan.csv
"""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek
from common_models import ridge

PHENO = "/home/nlapique/Downloads/BroadInst_DunedinPheno_20240829.txt"
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

ids = ek.dev_donors()
cov = ek.covariates(ids)
X = pd.concat([cov, ek.pseudobulk(ids, ["ALL"], min_cpm=3)], axis=1).reindex(ids)
Xc = cov.reindex(ids)
ph = pd.read_csv(PHENO, sep="\t", dtype={"snum": str}).set_index("snum")
ph = ph.apply(pd.to_numeric, errors="coerce").reindex(ids)
sex = ek.load().reindex(ids)["sex_M"].to_numpy(float)

# age-26 counterpart of each later measurement / slope / intercept
B26 = {"bmi": "bmi26", "map": "MeanArtPr26", "vox": "VOmax26", "chol": "choles26", "tri": "trigly26",
       "hdl": "hdl26", "urea": "urea26", "wbc": "wbc26"}
def baseline(t):
    m = {"BMI32": "bmi26", "bmi38": "bmi26", "bmip45": "bmi26", "Choles32": "choles26", "choles38": "choles26",
         "choles45": "choles26", "Trigly32": "trigly26", "trigly38": "trigly26", "trigly45": "trigly26",
         "HDL32": "hdl26", "HDL38": "hdl26", "HDL45": "hdl26", "Urea32": "urea26", "urea38": "urea26",
         "urea45": "urea26", "VOMax32": "VOmax26", "VOmax38": "VOmax26", "VOmax45": "VOmax26",
         "MeanArtPr32": "MeanArtPr26", "MeanArtPr38": "MeanArtPr26", "MeanArtPr45": "MeanArtPr26",
         "LnCRP32": "LnCRP26", "LnCRP38": "LnCRP26", "Ln_CRP45": "LnCRP26", "crp32": "crp26ns",
         "crp38": "crp26ns", "crp45": "crp26ns", "CurrentSmk32": "CurrentSmk26", "CurrentSmk38": "CurrentSmk26",
         "CurrentSmk45": "CurrentSmk26", "PackYrLifTm45": "PackYrsBy26"}
    if t in m:
        return m[t]
    if t.endswith(("38", "45")) and t[:-2] + "26" in ph.columns:        # DNAm cell fractions
        return t[:-2] + "26"
    if "_iz" in t:
        return B26.get(t.split("_")[0])
    return None

def cv_pred(Xm, y, Z=None):
    """OOF predictions (averaged over repeats) of y, or of y residualized on Z (fit in training fold)."""
    ok = ~np.isnan(y) & (np.ones(len(y), bool) if Z is None else ~np.isnan(Z).any(1))
    idx = np.where(ok)[0]
    rhos, oof = [], np.zeros(len(idx)); tgt_all = np.zeros(len(idx))
    for r, (tr, te) in enumerate(ek.splits(len(idx))):
        ytr, yte = y[idx[tr]], y[idx[te]]
        if Z is not None:
            A = np.c_[np.ones(len(tr)), Z[idx[tr]]]
            beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
            ytr = ytr - A @ beta
            yte = yte - np.c_[np.ones(len(te)), Z[idx[te]]] @ beta
        m = ridge()(); m.fit(Xm.iloc[idx[tr]], ytr)
        p = m.predict(Xm.iloc[idx[te]])
        rhos.append(stats.spearmanr(yte, p)[0] if np.std(p) > 0 else np.nan)
        oof[te] += p / ek.N_REPEATS
        if r < ek.N_FOLDS: tgt_all[te] = yte
    return np.nanmean(rhos), np.nanstd(rhos), ok.sum(), oof, y[idx]

def run(t):
    y = ph[t].to_numpy(float)
    if np.sum(~np.isnan(y)) < 300 or np.nanstd(y) == 0:
        return None
    out = {"trait": t, "n": int(np.sum(~np.isnan(y)))}
    out["rho_cov"], _, _, _, _ = cv_pred(Xc, y)
    out["rho_snRNA"], out["sd_snRNA"], _, oof, yy = cv_pred(X, y)
    if set(np.unique(yy)) <= {0, 1}:
        out["auc_snRNA"] = roc_auc_score(yy, oof)
    out["rho_sexadj"], out["sd_sexadj"], _, _, _ = cv_pred(X, y, sex[:, None])
    b = baseline(t)
    if b is not None and b != t:
        z = ph[b].to_numpy(float)
        out["baseline26"] = b
        ok = ~np.isnan(y) & ~np.isnan(z)
        out["rho_trait26"] = stats.spearmanr(y[ok], z[ok])[0]
        out["rho_b26adj"], out["sd_b26adj"], _, _, _ = cv_pred(X, y, np.c_[sex, z])
    out["rho_vs_pace"] = stats.spearmanr(y, ph["PaceOfAgingP45"], nan_policy="omit")[0]
    print(f"{t:22s} n={out['n']:4d} cov {out['rho_cov']:.2f} snRNA {out['rho_snRNA']:.2f} "
          f"sexadj {out['rho_sexadj']:.2f} b26adj {out.get('rho_b26adj', np.nan):.2f}", flush=True)
    return out

traits = sys.argv[2].split(",") if len(sys.argv) > 2 else [c for c in ph.columns if c != "sex"]
rows = Parallel(n_jobs=N_JOBS)(delayed(run)(t) for t in traits)
res = pd.DataFrame([r for r in rows if r]).sort_values("rho_sexadj", ascending=False)
if len(sys.argv) <= 2:
    res.to_csv(ek.RES + "trait_scan.csv", index=False)
print(res.round(3).to_string(index=False))
