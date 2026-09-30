"""Random-tertile permutation test for the Low-vs-High tertile gradient of the v11 scVI tertile sweep
(MULTISPLIT_PaceOfAgingP45_by_tertile2), for hdl26 and LnCRP26.

Question: is "AUC in the Low tertile minus AUC in the High tertile, averaged over the 9 cell types" larger than
what random groups of the same sizes give?

Pipeline = run_discovery_pdf_report (v2) from notebook (11) cell 58, exec'd unchanged except:
  * XGBoost on CPU (device="cpu", n_jobs=1) so that 14 combos run in parallel;
  * scVI-denoised expression is computed ONCE per cell (same model, library_size=1e4) and looked up, instead of
    being re-sampled inside every call (v11 draws one stochastic sample per call; the donor means average
    hundreds of cells, so this changes nothing systematic).
Settings as in the sweep: label_cutoffs="group", quantile_low=0.6, quantile_high=0.9, n_iterations=50,
stability 60%, top 1000 genes, both sexes pooled, outer 70/30 split with seeds 0..N_SPLITS-1.

perm 0 = the real tertiles (observed statistic, recomputed with this exact setup).
perm k>0 = the trait's values shuffled among donors that have one, then cut into tertiles (same group sizes).

Usage:  python permtest_tertile.py cache            # build the denoised-expression cache (GPU, ~1 min)
        python permtest_tertile.py run N_PERM        # run perms 0..N_PERM (resumes; results/rows.csv)
        python permtest_tertile.py summary
"""
import os, sys, json, time, itertools
import numpy as np, pandas as pd

BASE = "/home/nlapique/Blood/script/PERMTEST_tertile_scvi/"
H5 = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
MODEL = "/home/nlapique/Blood/model_scvi_adata_reduced_epochs_300_with_sex.h5ad"
NB11 = "/home/nlapique/Blood/script/notebooks/biomarker_v11_scvi_pseudobulk_ML.ipynb"
CACHE = BASE + "cache/denoised_f32.npy"
ROWS = BASE + "results/rows.csv"
TRAITS = ["hdl26", "LnCRP26"]
CELL_TYPES = ["B Memory", "B Naive", "CD4 Active", "CD8 Active", "Classical Monocytes", "MAIT cells",
              "NK cells", "Naive T cells", "Non-classical Monocytes"]
LEVELS = ["Low", "High"]
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
N_WORKERS = int(os.environ.get("N_WORKERS", 14))
os.makedirs(BASE + "cache", exist_ok=True); os.makedirs(BASE + "results", exist_ok=True)


def build_cache():
    import scanpy as sc, scvi
    from scvi.model import SCVI
    scvi.settings.seed = 0
    ad = sc.read_h5ad(H5)
    model = SCVI.load(MODEL, ad)
    D = np.lib.format.open_memmap(CACHE + ".tmp", mode="w+", dtype=np.float32, shape=ad.shape)
    for s in range(0, ad.n_obs, 50_000):
        e = min(s + 50_000, ad.n_obs)
        D[s:e] = model.get_normalized_expression(ad[s:e], library_size=1e4, return_numpy=True, batch_size=4096)
        print(f"  {e}/{ad.n_obs}", flush=True)
    D.flush(); del D
    os.replace(CACHE + ".tmp", CACHE)
    # slim AnnData for the workers: raw counts in X (used by v11 to rank candidate genes) + the obs it needs
    import anndata
    slim = anndata.AnnData(X=ad.X.tocsr(), obs=pd.DataFrame({
        "patient_id": ad.obs["patient_id"].astype(str).str.strip().values,
        "CellType": ad.obs["CellType"].astype(str).values,
        "cell_idx": np.arange(ad.n_obs)}, index=ad.obs_names), var=pd.DataFrame(index=ad.var_names))
    slim.write_h5ad(BASE + "cache/slim.h5ad")
    build_donors(ad.obs)
    print(f"cache done: {ad.shape}", flush=True)


def build_donors(obs):
    """All donors of `blood` with their traits (tertile cut-points use ALL of them, as in cell 71), plus the flag
    `in_analysis` = rows of df_pb_latent: >= 1 (donor, CellType) group with >= 50 cells and no NaN in BIO_COLS
    (sex, QC_MT.pct, PaceOfAgingP45, CurrentSmk26), as build_bio_indexed_from_blood + the inner join do."""
    obs = obs.copy(); obs["patient_id"] = obs["patient_id"].astype(str).str.strip()
    size = obs.groupby(["patient_id", "CellType"], observed=True).size()
    big = set(size[size >= 50].index.get_level_values(0))
    bio = ["sex", "QC_MT.pct", "PaceOfAgingP45", "CurrentSmk26"]
    tab = obs.drop_duplicates("patient_id").set_index("patient_id")[bio + TRAITS]
    tab["in_analysis"] = tab.index.isin(big) & tab[bio].notna().all(axis=1)
    tab.to_pickle(BASE + "cache/donors.pkl")
    print(f"{len(tab)} donors in blood, {tab.in_analysis.sum()} in the analysis", flush=True)


# ------------------------------------------------------------------ worker side
_W = {}


class CachedModel:
    """Stand-in for the scVI model: returns the cached denoised expression of the requested cells/genes."""
    def __init__(self, D, var_names):
        self.D, self.gpos = D, pd.Index(var_names)

    def get_normalized_expression(self, adata, gene_list=None, library_size=1e4, return_numpy=True, **kw):
        assert library_size == 1e4
        rows = adata.obs["cell_idx"].to_numpy()
        cols = self.gpos.get_indexer(gene_list)
        order = np.argsort(rows)                         # memmap reads are faster in row order
        out = np.empty((len(rows), len(cols)), np.float32)
        out[order] = self.D[rows[order]][:, cols]
        return out


def _init_worker():
    import anndata, json as _j
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from xgboost import XGBClassifier
    import xgboost as xgb
    ad = anndata.read_h5ad(BASE + "cache/slim.h5ad")
    D = np.load(CACHE, mmap_mode="r")
    src = "".join(_j.load(open(NB11))["cells"][58]["source"])
    n = src.count('device="cuda", tree_method="hist"')
    assert n == 4, n
    src = src.replace('device="cuda", tree_method="hist"', 'device="cpu", n_jobs=1, tree_method="hist"')
    g = dict(np=np, pd=pd, train_test_split=train_test_split, roc_curve=roc_curve, auc=auc,
             XGBClassifier=XGBClassifier, xgb=xgb, quantile_low=0.6, quantile_high=0.9)
    exec(src, g)
    _W.update(ad=ad, model=CachedModel(D, ad.var_names), fn=g["run_discovery_pdf_report"])


def permuted_categories(tab, trait, perm):
    s = tab[trait].copy()
    if perm > 0:
        ok = s.notna()
        rng = np.random.default_rng(10_000 * (TRAITS.index(trait) + 1) + perm)
        s[ok] = rng.permutation(s[ok].to_numpy())
    return pd.qcut(s, q=3, labels=["Low", "Medium", "High"])


def run_task(task):
    perm, trait, ct = task
    if not _W:
        _init_worker()
    tab = pd.read_pickle(BASE + "cache/donors.pkl")
    dfc = tab.copy()
    dfc[f"{trait}_cat"] = permuted_categories(tab, trait, perm)     # cut on all donors, as in cell 71
    dfc = dfc[dfc.in_analysis].drop(columns="in_analysis")
    dfc.index.name = "patient_id"
    out = []
    for lv in LEVELS:
        for seed in range(N_SPLITS):
            t0 = time.time()
            res = _W["fn"](model=_W["model"], adata=_W["ad"], df_clinical=dfc, biomarker="PaceOfAgingP45",
                           cell_type=ct, n_iterations=50, target_group=lv, group_col=f"{trait}_cat",
                           make_pdf=False, min_auc=0.0, outer_random_state=seed)
            out.append(dict(perm=perm, trait=trait, cell_type=ct, tertile=lv, seed=seed,
                            test_auc=np.nan if res is None else res["roc_auc"],
                            n_test=np.nan if res is None else res["n_test"], secs=round(time.time() - t0, 1)))
    return out


def run(n_perm):
    from multiprocessing import get_context
    done = set()
    if os.path.exists(ROWS):
        r = pd.read_csv(ROWS)
        cnt = r.groupby(["perm", "trait", "cell_type"]).size()
        done = set(cnt[cnt == len(LEVELS) * N_SPLITS].index)
    tasks = [t for t in itertools.product(range(n_perm + 1), TRAITS, CELL_TYPES) if t not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} done), {N_WORKERS} workers, {N_SPLITS} splits", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(N_WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_task, tasks), 1):
            pd.DataFrame(rows).to_csv(ROWS, mode="a", header=not os.path.exists(ROWS), index=False)
            if i % 18 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} tasks  {time.time()-t0:.0f}s", flush=True)
    summary()


def summary():
    r = pd.read_csv(ROWS)
    per = r.groupby(["perm", "trait", "cell_type", "tertile"]).test_auc.mean().unstack("tertile")
    per["diff"] = per["Low"] - per["High"]
    stat = per.groupby(["perm", "trait"])[["Low", "High", "diff"]].mean().reset_index()
    complete = per.groupby(["perm", "trait"]).size() == len(CELL_TYPES)
    stat = stat.set_index(["perm", "trait"])[complete].reset_index()
    lines = []
    for trait in TRAITS:
        s = stat[stat.trait == trait]
        if 0 not in s.perm.values:
            continue
        obs = s[s.perm == 0].iloc[0]; null = s[s.perm > 0]
        p = (1 + (null["diff"] >= obs["diff"]).sum()) / (1 + len(null))
        lines.append(f"{trait}: observed Low {obs.Low:.3f} / High {obs.High:.3f} / Low-High {obs['diff']:+.3f} | "
                     f"null ({len(null)} perms) Low-High mean {null['diff'].mean():+.3f}, SD {null['diff'].std():.3f}, "
                     f"95th pct {null['diff'].quantile(.95) if len(null) else np.nan:+.3f} | one-sided p = {p:.3f}")
        lines.append("   per cell type (observed): " + ", ".join(
            f"{ct}: {per.loc[(0, trait, ct), 'diff']:+.2f}" for ct in CELL_TYPES if (0, trait, ct) in per.index))
    txt = "\n".join(lines)
    open(BASE + "results/summary.txt", "w").write(txt + "\n")
    stat.to_csv(BASE + "results/stat_by_perm.csv", index=False)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        build_cache()
    elif cmd == "donors":
        import anndata
        build_donors(anndata.read_h5ad(H5, backed="r").obs)
    elif cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "summary":
        summary()
