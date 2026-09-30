"""Shared data loading, features and evaluation for the overnight PaceOfAgingP45 search.

Rules:
  * 20% of donors (data/lockbox.csv) are held out and NEVER used here; only final_lockbox.py touches them.
  * All exploration = repeated 5-fold CV on the remaining "dev" donors, same folds for every experiment.
  * Every supervised step (feature selection, scaling, imputation, tuning) happens inside the training fold.
  * Every experiment appends one line to results/results.csv and saves its out-of-fold predictions.
"""
import os
import re
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score, r2_score

BASE = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/"
DATA, RES = BASE + "data/", BASE + "results/"
TARGET = "PaceOfAgingP45"
Q_LOW, Q_HIGH = 0.6, 0.8          # High = top 20%, Low = bottom 60% (as in the notebook's GLOBAL CONFIG)
N_FOLDS, N_REPEATS = 5, 3

CLIN13 = ["bmi26", "choles26", "trigly26", "hdl26", "urea26", "VOmax26", "MeanArtPr26", "LnCRP26", "wbc26",
          "glyhaem26", "cortis26np", "CurrentSmk26", "PackYrsBy26"]
CLIN_EXTRA = ["astcur26", "RxBroncoDial26", "neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np",
              "HSV2_26"]
DNAM_CELLS = ["CD8T26", "CD4T26", "NK26", "Bcell26", "Mono26", "Gran26", "PlasmaBlast26",
              "CD8pCD28nCD45RAn_26", "CD8naive26", "CD4naive26"]
SMOKING = ["CurrentSmk26", "PackYrsBy26"]

_cache = {}


def load():
    if "donors" not in _cache:
        d = pd.read_pickle(DATA + "donors.pkl")
        d = d[d[TARGET].notna()].copy()
        _cache["donors"] = d
        _cache["groups"] = pd.read_pickle(DATA + "pb_groups.pkl")
        _cache["genes"] = np.array(open(DATA + "genes.txt").read().split("\n"))
    return _cache["donors"]


def aux_future():
    """Variables measured after age 26: ONLY for use as auxiliary training labels, never as predictors."""
    if "aux" not in _cache:
        _cache["aux"] = pd.read_pickle(DATA + "aux_future.pkl")
    return _cache["aux"]


def pb_arrays():
    if "pb_sum" not in _cache:
        _cache["pb_sum"] = np.load(DATA + "pb_sum.npy", mmap_mode="r")
        _cache["pb_det"] = np.load(DATA + "pb_det.npy", mmap_mode="r")
    return _cache["pb_sum"], _cache["pb_det"], _cache["groups"], _cache["genes"]


def make_lockbox(frac=0.2, seed=2026):
    """Stratified (by pace decile) random 20% of donors, written once and never regenerated."""
    path = DATA + "lockbox.csv"
    if os.path.exists(path):
        return
    d = load()
    rng = np.random.default_rng(seed)
    dec = pd.qcut(d[TARGET], 10, labels=False)
    lock = []
    for k in range(10):
        ids = d.index[dec == k].to_numpy()
        lock += list(rng.choice(ids, size=int(round(frac * len(ids))), replace=False))
    pd.Series(sorted(lock), name="patient_id").to_csv(path, index=False)


def dev_donors():
    d = load()
    lock = set(pd.read_csv(DATA + "lockbox.csv", dtype=str)["patient_id"])
    return d.index[~d.index.isin(lock)]


def lock_donors():
    return pd.Index(pd.read_csv(DATA + "lockbox.csv", dtype=str)["patient_id"])


# ------------------------------------------------------------------ features
def covariates(ids):
    d = load().reindex(ids)
    return pd.DataFrame({"cov|sex_M": d["sex_M"], "cov|log_ncells": np.log(d["n_cells"]),
                         "cov|log_umi": np.log(d["mean_umi"]), "cov|pct_mt": d["mean_pct_mt"]}, index=ids)


def clinical(ids, cols=CLIN13):
    return load().reindex(ids)[cols].apply(pd.to_numeric, errors="coerce").add_prefix("clin|")


def composition(ids, which="celltype", pseudo=0.5):
    c = pd.read_pickle(DATA + f"comp_{which}.pkl").reindex(ids).fillna(0)
    lp = np.log(c + pseudo)
    return lp.sub(lp.mean(axis=1), axis=0).add_prefix(f"clr_{which}|")


def pseudobulk(ids, cell_types=("ALL",), min_cells=20, min_frac_donors=0.5, min_cpm=10.0, kind="logcpm",
               gene_regex=None, genes=None, top_var=None):
    """Donor x (cell type | gene) matrix. kind: logcpm | det (fraction of cells expressing).
    Gene filtering is label-free (expression level only), so it can be done on all dev donors."""
    S, D, gi, gn = pb_arrays()
    out = []
    for ct in cell_types:
        if ct == "ALL":
            rows = gi.index.to_numpy()
            g2 = gi.assign(r=rows)
            per = {}
            for pid, sub in g2.groupby("patient_id"):
                per[pid] = sub["r"].to_numpy()
            pids = [p for p in ids if p in per]
            s = np.stack([np.asarray(S[per[p]]).sum(0) for p in pids])
            dt = np.stack([np.asarray(D[per[p]]).sum(0) for p in pids])
            nc = np.array([gi.loc[per[p], "n_cells"].sum() for p in pids])
        else:
            sub = gi[(gi["CellType"] == ct) & (gi["n_cells"] >= min_cells) & gi["patient_id"].isin(ids)]
            pids = sub["patient_id"].tolist()
            s = np.asarray(S[sub.index.to_numpy()])
            dt = np.asarray(D[sub.index.to_numpy()])
            nc = sub["n_cells"].to_numpy()
        if len(pids) < min_frac_donors * len(ids):
            continue
        cpm = s / s.sum(1, keepdims=True) * 1e6
        keep = cpm.mean(0) >= min_cpm
        if genes is not None:
            keep &= np.isin(gn, list(genes))
        if gene_regex is not None:
            keep &= np.array([bool(re.search(gene_regex, g)) for g in gn])
        if kind == "logcpm":
            m = np.log1p(cpm[:, keep])
        elif kind == "det":
            m = dt[:, keep] / nc[:, None]
        else:
            raise ValueError(kind)
        cols = gn[keep]
        if top_var is not None and m.shape[1] > top_var:
            v = m.var(0)
            idx = np.argsort(v)[::-1][:top_var]
            m, cols = m[:, idx], cols[idx]
        out.append(pd.DataFrame(m, index=pids, columns=[f"pb|{ct}|{g}" for g in cols]))
    return pd.concat(out, axis=1).reindex(ids)


# ------------------------------------------------------------------ evaluation
def splits(n):
    return list(RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=0).split(np.arange(n)))


def _fold_metrics(y_tr, y_te, pred):
    lo, hi = np.quantile(y_tr, Q_LOW), np.quantile(y_tr, Q_HIGH)
    ext = (y_te <= lo) | (y_te >= hi)
    lab = (y_te[ext] >= hi).astype(int)
    auc = roc_auc_score(lab, pred[ext]) if 0 < lab.sum() < len(lab) else np.nan
    rho = stats.spearmanr(y_te, pred)[0] if np.std(pred) > 0 else np.nan
    return auc, rho


def evaluate(name, X, model_factory, task="reg", y=None, notes="", save=True, verbose=True):
    """X: DataFrame indexed by dev donors (same order as dev_donors()). model_factory() -> sklearn estimator.
    task: 'reg' = regress continuous pace; 'clf' = classify High vs Low (train-fold extremes only, middle dropped)
    and score with predict_proba; 'clf_top' = High vs rest."""
    ids = dev_donors()
    X = X.reindex(ids)
    yv = (load().reindex(ids)[TARGET] if y is None else y.reindex(ids)).to_numpy(dtype=float)
    t0 = time.time()
    oof = np.full((N_REPEATS, len(ids)), np.nan)
    aucs, rhos, r2s = [], [], []
    for k, (tr, te) in enumerate(splits(len(ids))):
        m = model_factory()
        if task == "reg":
            m.fit(X.iloc[tr], yv[tr])
            p = m.predict(X.iloc[te])
            r2s.append(r2_score(yv[te], p))
        else:
            lo, hi = np.quantile(yv[tr], Q_LOW), np.quantile(yv[tr], Q_HIGH)
            use = (yv[tr] <= lo) | (yv[tr] >= hi) if task == "clf" else np.ones(len(tr), bool)
            m.fit(X.iloc[tr[use]], (yv[tr[use]] >= hi).astype(int))
            p = m.predict_proba(X.iloc[te])[:, 1]
        oof[k // N_FOLDS, te] = p
        a, r = _fold_metrics(yv[tr], yv[te], p)
        aucs.append(a); rhos.append(r)
    res = {"name": name, "task": task, "n_features": X.shape[1], "auc_mean": np.nanmean(aucs),
           "auc_sd": np.nanstd(aucs), "spearman_mean": np.nanmean(rhos), "spearman_sd": np.nanstd(rhos),
           "r2_mean": np.mean(r2s) if r2s else np.nan, "secs": round(time.time() - t0, 1), "notes": notes,
           "time": time.strftime("%Y-%m-%d %H:%M")}
    if save:
        path = RES + "results.csv"
        pd.DataFrame([res]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        np.savez_compressed(RES + f"oof/{safe(name)}.npz", oof=oof, aucs=np.array(aucs), rhos=np.array(rhos),
                            ids=np.array(ids))
    if verbose:
        print(f"{name:70s} AUC {res['auc_mean']:.3f}±{res['auc_sd']:.3f}  rho {res['spearman_mean']:.3f}"
              f"  R2 {res['r2_mean']:.3f}  p={X.shape[1]}  {res['secs']}s", flush=True)
    return res


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)[:150]


def compare(name_a, name_b):
    """Paired corrected t-test (Nadeau-Bengio) on per-fold AUC between two saved experiments."""
    a = np.load(RES + f"oof/{safe(name_a)}.npz")["aucs"]
    b = np.load(RES + f"oof/{safe(name_b)}.npz")["aucs"]
    d = a - b
    d = d[~np.isnan(d)]
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + 1 / (N_FOLDS - 1))
    t = d.mean() / np.sqrt(var)
    return d.mean(), 2 * stats.t.sf(abs(t), J - 1)


def pseudobulk_libcorr(ids, cell_type="ALL", min_cpm=3.0, ref_ids=None, prior=1.0, weight="ncells",
                       min_group_cells=3):
    """Library-corrected pseudobulk: log-CPM per (donor, library[, cell type]) group, minus the library's mean
    profile (estimated on ref_ids = dev donors only), then a cell-count-weighted average per donor. Label-free."""
    name = "lib" if cell_type == "ALL" else "libct"
    key = f"pb_{name}"
    if key not in _cache:
        _cache[key] = (np.load(DATA + f"{key}_sum.npy", mmap_mode="r"), pd.read_pickle(DATA + f"{key}_groups.pkl"))
    S, gi = _cache[key]
    gn = pb_arrays()[3]
    ref = set(dev_donors() if ref_ids is None else ref_ids)
    sub = gi[(gi["CellType"] == cell_type) & (gi["n_cells"] >= min_group_cells)
             & (gi["patient_id"].isin(set(ids) | ref))]
    s = np.asarray(S[sub.index.to_numpy()], dtype=np.float64)
    lib_size = s.sum(1, keepdims=True)
    cpm_all = s.sum(0) / s.sum() * 1e6
    keep = cpm_all >= min_cpm
    lc = np.log((s[:, keep] + prior) / (lib_size + prior) * 1e6)
    w = sub["n_cells"].to_numpy(float) if weight == "ncells" else np.ones(len(sub))
    in_ref = sub["patient_id"].isin(ref).to_numpy()
    libs = sub["library"].to_numpy()
    corr = lc.copy()
    grand = np.average(lc[in_ref], axis=0, weights=w[in_ref])
    for L in np.unique(libs):
        m = (libs == L)
        mr = m & in_ref
        if mr.sum() >= 3:
            corr[m] -= np.average(lc[mr], axis=0, weights=w[mr]) - grand
    df = pd.DataFrame(corr, columns=[f"pbL|{cell_type}|{g}" for g in gn[keep]])
    df["__pid"] = sub["patient_id"].to_numpy(); df["__w"] = w
    out = df.groupby("__pid").apply(lambda t: pd.Series(np.average(t.drop(columns=["__pid", "__w"]).to_numpy(),
                                                                    axis=0, weights=t["__w"].to_numpy()),
                                                         index=t.columns[:-2]))
    return out.reindex(ids)


def evaluate_blend(label, names, weights=None, save=True):
    """Average of saved out-of-fold predictions (z-scored within each test fold), scored on the same folds.
    A fixed rule, no fitting; choosing WHICH models to blend is still a selection step -> lock-box decides."""
    ids = dev_donors()
    y = load().reindex(ids)[TARGET].to_numpy(float)
    P = [np.load(RES + f"oof/{safe(n)}.npz", allow_pickle=True) for n in names]
    for p in P:
        assert list(p["ids"]) == list(ids)
    w = np.ones(len(names)) if weights is None else np.asarray(weights, float)
    aucs, rhos = [], []
    oof = np.full((N_REPEATS, len(ids)), np.nan)
    for k, (tr, te) in enumerate(splits(len(ids))):
        r = k // N_FOLDS
        z = [(p["oof"][r, te] - p["oof"][r, te].mean()) / (p["oof"][r, te].std() + 1e-12) for p in P]
        pred = np.average(np.vstack(z), axis=0, weights=w)
        oof[r, te] = pred
        a, rh = _fold_metrics(y[tr], y[te], pred)
        aucs.append(a); rhos.append(rh)
    res = {"name": label, "task": "blend", "n_features": len(names), "auc_mean": np.nanmean(aucs),
           "auc_sd": np.nanstd(aucs), "spearman_mean": np.nanmean(rhos), "spearman_sd": np.nanstd(rhos),
           "r2_mean": np.nan, "secs": 0, "notes": " + ".join(names), "time": time.strftime("%Y-%m-%d %H:%M")}
    if save:
        pd.DataFrame([res]).to_csv(RES + "results.csv", mode="a", header=False, index=False)
        np.savez_compressed(RES + f"oof/{safe(label)}.npz", oof=oof, aucs=np.array(aucs), rhos=np.array(rhos),
                            ids=np.array(ids))
    print(f"{label:70s} AUC {res['auc_mean']:.3f}±{res['auc_sd']:.3f}  rho {res['spearman_mean']:.3f}", flush=True)
    return res


def lognorm_moments(ids, cell_type="ALL", kind="mean", min_cells=20, min_cpm=3.0):
    """Per donor (x cell type): mean or variance across cells of log1p(CP10k). Genes filtered by pseudobulk CPM."""
    if "ln1" not in _cache:
        _cache["ln1"] = np.load(DATA + "ln_sum.npy", mmap_mode="r")
        _cache["ln2"] = np.load(DATA + "ln_sumsq.npy", mmap_mode="r")
    S, D, gi, gn = pb_arrays()
    L1, L2 = _cache["ln1"], _cache["ln2"]
    if cell_type == "ALL":
        per = {p: sub.index.to_numpy() for p, sub in gi.groupby("patient_id")}
        pids = [p for p in ids if p in per]
        agg = lambda A: np.stack([np.asarray(A[per[p]]).sum(0) for p in pids])
        s1, s2, s0 = agg(L1), agg(L2), agg(S)
        nc = np.array([gi.loc[per[p], "n_cells"].sum() for p in pids], float)
    else:
        sub = gi[(gi["CellType"] == cell_type) & (gi["n_cells"] >= min_cells) & gi["patient_id"].isin(ids)]
        pids = sub["patient_id"].tolist(); r = sub.index.to_numpy()
        s1, s2, s0 = np.asarray(L1[r]), np.asarray(L2[r]), np.asarray(S[r]); nc = sub["n_cells"].to_numpy(float)
    cpm = s0.sum(0) / s0.sum() * 1e6
    keep = cpm >= min_cpm
    mu = s1[:, keep] / nc[:, None]
    if kind == "mean":
        m = mu
    elif kind == "var":
        m = np.maximum(s2[:, keep] / nc[:, None] - mu ** 2, 0) * nc[:, None] / np.maximum(nc[:, None] - 1, 1)
    else:
        raise ValueError(kind)
    return pd.DataFrame(m, index=pids, columns=[f"ln{kind}|{cell_type}|{g}" for g in gn[keep]]).reindex(ids)
