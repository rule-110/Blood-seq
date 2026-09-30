"""Build donor-level caches from the snRNA-seq h5ad files (run once).

Outputs in ../data:
  donors.pkl          one row per donor: age-26 clinical, targets, sex, batch, QC summaries
  pb_sum.npy          (groups x genes) summed raw counts, full transcriptome, per donor x CellType + ALL
  pb_det.npy          (groups x genes) number of cells with count > 0
  pb_groups.pkl       index of the rows above (patient_id, CellType) and n_cells
  genes.txt           gene names (columns)
  comp_*.pkl          cell counts per donor x (CellType / leiden_0.5 / leiden_1.0)
"""
import time
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem

CB = "/home/nlapique/Blood/cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad"
RD = "/home/nlapique/Blood/adata_reduced_HVG_2000_with_sex_qcmt.h5ad"
OUT = "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/"

CLIN26 = ["bmi26", "choles26", "trigly26", "hdl26", "urea26", "VOmax26", "MeanArtPr26", "LnCRP26", "wbc26",
          "glyhaem26", "cortis26np", "CurrentSmk26", "PackYrsBy26", "astcur26", "RxBroncoDial26",
          "neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np", "HSV2_26",
          "CD8T26", "CD4T26", "NK26", "Bcell26", "Mono26", "Gran26", "PlasmaBlast26",
          "CD8pCD28nCD45RAn_26", "CD8naive26", "CD4naive26"]
TARGETS = ["PaceOfAgingP45"] + [f"POA45_no{s}" for s in
           ["bmi", "wh", "gly", "lep", "map", "vox", "fvc", "fev", "chol", "tri", "hdl", "lipo", "apol",
            "egfr", "urea", "crp", "wbc", "cal", "rcor"]]

t0 = time.time()
with h5py.File(CB) as f:
    obs = pd.DataFrame({c: np.asarray(read_elem(f["obs"][c])) for c in
                        ["patient_id", "sex", "batch_merging" if "batch_merging" in f["obs"] else "sample",
                         "total_counts", "pct_counts_mt", "n_genes_by_counts", "doublet_score", "leiden_0.5",
                         "leiden_1.0", *CLIN26, *TARGETS] if c in f["obs"]})
    gene_names = np.asarray(read_elem(f["var"]).index).astype(str)
with h5py.File(RD) as g:
    obs["CellType"] = np.asarray(read_elem(g["obs"]["CellType"])).astype(str)
    obs["batch"] = np.asarray(read_elem(g["obs"]["batch_merging"])).astype(str)
obs["patient_id"] = obs["patient_id"].astype(str).str.strip()
print(f"obs read: {obs.shape} in {time.time() - t0:.0f}s", flush=True)

# ---- donor table ----
grp = obs.groupby("patient_id")
donors = grp[[*CLIN26, *TARGETS]].first()
donors["sex_M"] = grp["sex"].first().astype(str).str.replace(".0", "", regex=False).map(
    {"1": 0.0, "2": 1.0, "F": 0.0, "M": 1.0})
donors["n_cells"] = grp.size()
donors["mean_umi"] = grp["total_counts"].mean()
donors["mean_genes"] = grp["n_genes_by_counts"].mean()
donors["mean_pct_mt"] = grp["pct_counts_mt"].mean()
donors["mean_doublet"] = grp["doublet_score"].mean()
bt = obs.groupby(["patient_id", "batch"]).size().rename("n").reset_index().sort_values("n", ascending=False)
bt = bt.drop_duplicates("patient_id").set_index("patient_id")
donors["batch_major"] = bt["batch"]
donors["batch_major_frac"] = bt["n"] / donors["n_cells"]
donors.to_pickle(OUT + "donors.pkl")
print(donors[["sex_M", "n_cells", "batch_major", "batch_major_frac"]].describe(include="all").T, flush=True)

for col, name in [("CellType", "celltype"), ("leiden_0.5", "leiden05"), ("leiden_1.0", "leiden10")]:
    pd.crosstab(obs["patient_id"], obs[col].astype(str)).to_pickle(OUT + f"comp_{name}.pkl")

# ---- pseudobulk: donor x CellType, full transcriptome ----
keys = pd.MultiIndex.from_arrays([obs["patient_id"].values, obs["CellType"].values], names=["patient_id", "CellType"])
codes, groups = pd.factorize(keys)
n = len(codes)
G = sparse.csc_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(groups), n))
CH = 40_000
S = None
D = None
with h5py.File(CB) as f:
    L = f["layers"]["counts"]
    ip = L["indptr"][:]
    ng = int(L.attrs["shape"][1])
    for s in range(0, n, CH):
        e = min(s + CH, n)
        a, b = ip[s], ip[e]
        X = sparse.csr_matrix((L["data"][a:b].astype(np.float32), L["indices"][a:b], ip[s:e + 1] - a), shape=(e - s, ng))
        Gs = G[:, s:e]
        ps = (Gs @ X)
        Xb = X.copy(); Xb.data[:] = 1.0
        pd_ = (Gs @ Xb)
        S = ps if S is None else S + ps
        D = pd_ if D is None else D + pd_
        print(f"   cells {e}/{n}  ({time.time() - t0:.0f}s)", flush=True)
S = S.toarray().astype(np.float32)
D = D.toarray().astype(np.float32)
np.save(OUT + "pb_sum.npy", S)
np.save(OUT + "pb_det.npy", D)
gi = pd.DataFrame(list(groups), columns=["patient_id", "CellType"])
gi["n_cells"] = np.bincount(codes)
gi.to_pickle(OUT + "pb_groups.pkl")
with open(OUT + "genes.txt", "w") as fh:
    fh.write("\n".join(gene_names))
print(f"done: pb_sum {S.shape} in {time.time() - t0:.0f}s", flush=True)
