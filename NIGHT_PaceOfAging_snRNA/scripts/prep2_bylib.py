"""Pseudobulk per (donor, library) and per (donor, library, CellType) for the main cell types, full transcriptome."""
import time, h5py, numpy as np, pandas as pd
from scipy import sparse
try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem
CB = "/home/nlapique/Blood/cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad"
RD = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
OUT = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/"
t0 = time.time()
with h5py.File(RD) as g:
    pid = np.asarray(read_elem(g["obs"]["patient_id"])).astype(str)
    lib = np.asarray(read_elem(g["obs"]["batch_merging"])).astype(str)
    ct = np.asarray(read_elem(g["obs"]["CellType"])).astype(str)
pid = np.char.strip(pid)
for name, keys_arr in [("lib", [pid, lib, np.array(["ALL"] * len(pid))]), ("libct", [pid, lib, ct])]:
    keys = pd.MultiIndex.from_arrays(keys_arr, names=["patient_id", "library", "CellType"])
    codes, groups = pd.factorize(keys)
    n = len(codes)
    G = sparse.csc_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(groups), n))
    S = None
    with h5py.File(CB) as f:
        L = f["layers"]["counts"]; ip = L["indptr"][:]; ng = int(L.attrs["shape"][1])
        for s in range(0, n, 40_000):
            e = min(s + 40_000, n); a, b = ip[s], ip[e]
            X = sparse.csr_matrix((L["data"][a:b].astype(np.float32), L["indices"][a:b], ip[s:e + 1] - a), shape=(e - s, ng))
            ps = G[:, s:e] @ X
            S = ps if S is None else S + ps
    np.save(OUT + f"pb_{name}_sum.npy", S.toarray().astype(np.float32))
    gi = pd.DataFrame(list(groups), columns=["patient_id", "library", "CellType"]); gi["n_cells"] = np.bincount(codes)
    gi.to_pickle(OUT + f"pb_{name}_groups.pkl")
    print(name, len(groups), f"{time.time()-t0:.0f}s", flush=True)
