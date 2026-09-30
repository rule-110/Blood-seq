"""Per (donor, CellType) first and second moments of log1p(CP10k) per cell, full transcriptome.
Gives mean-of-log expression and within-donor cell-to-cell variance."""
import time, h5py, numpy as np, pandas as pd
from scipy import sparse
CB = "/home/nlapique/Blood/cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad"
OUT = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/"
gi = pd.read_pickle(OUT + "pb_groups.pkl")
import sys; sys.path.insert(0, OUT + "../scripts")
from anndata.io import read_elem
RD = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
with h5py.File(RD) as g:
    pid = np.char.strip(np.asarray(read_elem(g["obs"]["patient_id"])).astype(str))
    ct = np.asarray(read_elem(g["obs"]["CellType"])).astype(str)
lut = {(p, c): i for i, (p, c) in enumerate(zip(gi["patient_id"], gi["CellType"]))}
codes = np.array([lut[(p, c)] for p, c in zip(pid, ct)])
n = len(codes)
G = sparse.csc_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(gi), n))
t0 = time.time(); S1 = S2 = None
with h5py.File(CB) as f:
    L = f["layers"]["counts"]; ip = L["indptr"][:]; ng = int(L.attrs["shape"][1])
    for s in range(0, n, 40_000):
        e = min(s + 40_000, n); a, b = ip[s], ip[e]
        X = sparse.csr_matrix((L["data"][a:b].astype(np.float32), L["indices"][a:b], ip[s:e + 1] - a), shape=(e - s, ng))
        lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
        X = sparse.diags(1e4 / lib) @ X
        X.data = np.log1p(X.data)
        X2 = X.copy(); X2.data **= 2
        Gs = G[:, s:e]
        a1, a2 = Gs @ X, Gs @ X2
        S1 = a1 if S1 is None else S1 + a1
        S2 = a2 if S2 is None else S2 + a2
np.save(OUT + "ln_sum.npy", S1.toarray().astype(np.float32))
np.save(OUT + "ln_sumsq.npy", S2.toarray().astype(np.float32))
print(f"done {time.time()-t0:.0f}s", flush=True)
