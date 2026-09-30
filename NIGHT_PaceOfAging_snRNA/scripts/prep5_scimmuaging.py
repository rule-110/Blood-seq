"""sc-ImmuAging clocks (Li et al., Nat Aging 2025) applied to our nuclei.

Clock = LASSO on Seurat log-normalized expression (log1p(counts / library * 1e4)). The original code averages
100 pseudocells of 15 random cells per donor and cell type and predicts on each. The clock is linear, so the
mean over pseudocells equals (in expectation) the prediction on the donor's mean log-normalized profile; we use
that exact mean instead of random pseudocells.

Output (data/scimmuaging_groups.pkl): one row per (donor, library, clock cell type) with n_cells and the
clock score of the group's mean profile. Library = obs batch_merging, full-transcriptome library size.
Cell-type mapping: CD4T = CD4 Active + Naive T (CD8A/CD8B = 0); CD8T = CD8 Active + Naive T (CD8A or CD8B > 0);
MONO = Classical + Non-classical Monocytes; NK = NK cells; B = B Naive + B Memory. MAIT / Plasma not used.
"""
import time, h5py, numpy as np, pandas as pd
from scipy import sparse
from anndata.io import read_elem

CB = "/home/nlapique/Blood/cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad"
RD = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
OUT = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/"
COEF = "/home/nlapique/Blood/script/scImmuAging_clock/clock_coefs_symbols.csv"
t0 = time.time()

coef = pd.read_csv(COEF)
icpt = coef[coef.gene == "(Intercept)"].set_index("cell_type")["coef"]
coef = coef[coef.gene != "(Intercept)"].dropna(subset=["symbol"])

with h5py.File(RD) as g:
    pid = np.char.strip(np.asarray(read_elem(g["obs"]["patient_id"])).astype(str))
    lib = np.asarray(read_elem(g["obs"]["batch_merging"])).astype(str)
    ct = np.asarray(read_elem(g["obs"]["CellType"])).astype(str)
with h5py.File(CB) as f:
    var = pd.Index(np.asarray(read_elem(f["var"])).astype(str) if False else read_elem(f["var"]).index.astype(str))
    pid_cb = np.char.strip(np.asarray(read_elem(f["obs"]["patient_id"])).astype(str))
assert (pid_cb == pid).all(), "cell order differs between the two h5ad files"

genes = sorted(set(coef.symbol) & set(var)) + ["CD4", "CD8A", "CD8B"]
gidx = var.get_indexer(genes)
assert (gidx >= 0).all()
print(f"{len(genes)} genes to extract", flush=True)

# --- log-normalized expression of the clock genes for every cell (full-transcriptome library) ---
blocks = []
with h5py.File(CB) as f:
    L = f["layers"]["counts"]; ip = L["indptr"][:]; ng = int(L.attrs["shape"][1]); n = len(ip) - 1
    for s in range(0, n, 40_000):
        e = min(s + 40_000, n); a, b = ip[s], ip[e]
        X = sparse.csr_matrix((L["data"][a:b].astype(np.float32), L["indices"][a:b], ip[s:e + 1] - a),
                              shape=(e - s, ng))
        libsize = np.asarray(X.sum(1)).ravel(); libsize[libsize == 0] = 1
        X = X[:, gidx]
        X = sparse.diags((1e4 / libsize).astype(np.float32)) @ X
        X.data = np.log1p(X.data)
        blocks.append(X.tocsr())
        print(f"  cells {e}/{n}  {time.time()-t0:.0f}s", flush=True)
E = sparse.vstack(blocks).tocsr()
gpos = {g: i for i, g in enumerate(genes)}

# --- clock cell types ---
cd8 = (E[:, gpos["CD8A"]].toarray().ravel() > 0) | (E[:, gpos["CD8B"]].toarray().ravel() > 0)
clock_ct = np.full(len(ct), "", dtype=object)
clock_ct[ct == "CD4 Active"] = "CD4T"
clock_ct[ct == "CD8 Active"] = "CD8T"
clock_ct[(ct == "Naive T cells") & ~cd8] = "CD4T"
clock_ct[(ct == "Naive T cells") & cd8] = "CD8T"
clock_ct[np.isin(ct, ["Classical Monocytes", "Non-classical Monocytes"])] = "MONO"
clock_ct[ct == "NK cells"] = "NK"
clock_ct[np.isin(ct, ["B Naive", "B Memory"])] = "B"
print("Naive T cells split: CD8", int(((ct == "Naive T cells") & cd8).sum()), "| CD4", int(((ct == "Naive T cells") & ~cd8).sum()))
print(pd.Series(clock_ct).value_counts().to_string(), flush=True)

# --- group means per (donor, library, clock cell type) and clock scores ---
use = clock_ct != ""
keys = pd.MultiIndex.from_arrays([pid[use], lib[use], clock_ct[use].astype(str)],
                                 names=["patient_id", "library", "clock_ct"])
codes, groups = pd.factorize(keys)
m = len(codes)
G = sparse.csr_matrix((np.ones(m, np.float32), (codes, np.arange(m))), shape=(len(groups), m))
ncell = np.asarray(G.sum(1)).ravel()
M = np.asarray((G @ E[use]).todense()) / ncell[:, None]          # group mean log-normalized profile
gi = pd.DataFrame(list(groups), columns=["patient_id", "library", "clock_ct"]); gi["n_cells"] = ncell.astype(int)
gi["score"] = np.nan
for c in icpt.index:
    cc = coef[(coef.cell_type == c) & coef.symbol.isin(genes)]
    w = np.zeros(len(genes)); w[[gpos[s] for s in cc.symbol]] = cc.coef.to_numpy()
    r = gi.clock_ct == c
    gi.loc[r, "score"] = M[r.to_numpy()] @ w + icpt[c]
gi.to_pickle(OUT + "scimmuaging_groups.pkl")
print(gi.groupby("clock_ct")["score"].describe().round(1).to_string())
print(f"done {time.time()-t0:.0f}s", flush=True)
