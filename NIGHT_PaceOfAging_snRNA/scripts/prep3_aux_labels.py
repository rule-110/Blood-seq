"""Future-measured variables used ONLY as auxiliary training labels (never as predictors)."""
import h5py, numpy as np, pandas as pd
try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem
CB = "/home/nlapique/Blood/cell_bender_processed/combined_annotated_final_with_dunedinpace_traits.h5ad"
COLS = ["bmi_iz_snL", "map_iz_snL", "vox_izr_snL", "chol_iz_snL", "tri_iz_snL", "hdl_izr_snL", "urea_iz_snL",
        "wbc_iz_snL", "suPAR45", "crp32", "crp38", "crp45", "BMI32", "bmi38", "bmip45", "HDL32", "HDL38", "HDL45",
        "trigly45", "choles45", "urea45", "VOmax45", "MeanArtPr45", "Ln_CRP45", "CurrentSmk45", "PackYrLifTm45",
        "CD8pCD28nCD45RAn_38", "CD8pCD28nCD45RAn_45"]
with h5py.File(CB) as f:
    obs = pd.DataFrame({c: np.asarray(read_elem(f["obs"][c])) for c in ["patient_id", *COLS]})
obs["patient_id"] = obs["patient_id"].astype(str).str.strip()
aux = obs.groupby("patient_id")[COLS].first().apply(pd.to_numeric, errors="coerce")
aux.to_pickle("/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/data/aux_future.pkl")
print(aux.notna().mean().round(2).to_string())
