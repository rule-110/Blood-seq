# v11 tertile sweep WITHOUT scVI: identical to the tertile sweep cell of
# "Biomarker_prediction_scVI_pseudobulk_ML_h5ad_direct (11).ipynb", except that donor features are the mean of
# normalized raw counts (CP10k) instead of scVI-denoised expression.
#
# How: run_discovery_pdf_report() only uses scVI in one line,
#     model.get_normalized_expression(adata_grp, gene_list=..., library_size=1e4, return_numpy=True)
# so we pass it a stand-in "model" whose get_normalized_expression returns counts / total_counts * 1e4
# for the same cells and genes. Labels, splits, candidate genes, stability selection, inner validation,
# final XGBoost and AUCs all run through the unchanged v11 functions.
#
# Run it from a notebook cell placed AFTER the tertile-category cell (it uses the kernel's variables:
# adata, df_pb_latent, TRAITS_FOR_TERTILE, my_valid_types, run_discovery_multi_split, BIOMARKER, N_ITER_PDF):
#     %run -i v11_counts_tertile_sweep.py
# Settings can be overridden by defining them in the notebook before the %run line, e.g.
#     CELL_TYPES_FOR_TERTILE_COUNTS = ["CD4 Active", "B Naive"]
#
# PROGRESSIVE SAVE + RESUME: same as the scVI cell. After every (cell_type, trait, tertile) combo the summary
# CSV + Excel workbook are rewritten and per-split AUCs / gene hits go to per_combo/. After a kernel crash,
# re-run the %run line: finished combos are loaded from disk and skipped.
import os, re, gc
import numpy as np
import pandas as pd
from scipy import sparse


class NormalizedCountsModel:
    """Stand-in for the scVI model: same get_normalized_expression() call, returns normalized raw counts."""

    def __init__(self, layer="counts", library_col="total_counts"):
        self.layer = layer                # raw integer counts (layers["counts"]; .X holds the same values)
        self.library_col = library_col    # per-cell library size over the FULL transcriptome (obs["total_counts"])

    def get_normalized_expression(self, adata, gene_list=None, library_size=1e4, return_numpy=True, **kwargs):
        X = adata.layers[self.layer] if self.layer in adata.layers else adata.X
        idx = adata.var_names.get_indexer(gene_list) if gene_list is not None else np.arange(adata.n_vars)
        if (idx < 0).any():
            raise KeyError("some genes of gene_list are not in adata.var_names")
        lib = pd.to_numeric(adata.obs[self.library_col], errors="coerce").to_numpy(dtype=float)
        X = X[:, idx]
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        out = X.astype(np.float32) / lib[:, None].astype(np.float32) * np.float32(library_size)
        return out if return_numpy else pd.DataFrame(out, index=adata.obs_names, columns=list(gene_list))


COUNTS_MODEL = NormalizedCountsModel()

# --- settings (defaults = same as the scVI tertile cell; define any of these before %run to override) ---
CELL_TYPES_FOR_TERTILE_COUNTS = globals().get("CELL_TYPES_FOR_TERTILE_COUNTS", my_valid_types)
N_OUTER_SPLITS_TERTILE_COUNTS = globals().get("N_OUTER_SPLITS_TERTILE_COUNTS", 20)
RESUME_TERTILE_COUNTS = globals().get("RESUME_TERTILE_COUNTS", True)
TERTILE_LEVELS = ["Low", "Medium", "High"]

OUTDIR_TERTILE_COUNTS = f"MULTISPLIT_{BIOMARKER}_by_tertile2_counts"
COMBO_DIR_TERTILE_COUNTS = f"{OUTDIR_TERTILE_COUNTS}/per_combo"
SUMMARY_TERTILE_COUNTS_CSV = f"{OUTDIR_TERTILE_COUNTS}/summary_all_celltypes_by_tertile_counts.csv"
SUMMARY_TERTILE_COUNTS_XLSX = f"{OUTDIR_TERTILE_COUNTS}/RESULTATS_TERTILE_COUNTS_{BIOMARKER}.xlsx"
CRASH_TERTILE_COUNTS_CSV = f"{OUTDIR_TERTILE_COUNTS}/crashes_by_tertile.csv"
os.makedirs(COMBO_DIR_TERTILE_COUNTS, exist_ok=True)

_missing_cat = [t for t in TRAITS_FOR_TERTILE if f"{t}_cat" not in df_pb_latent.columns]
assert len(_missing_cat) < len(TRAITS_FOR_TERTILE), (
    "No tertile category columns in df_pb_latent -- run the tertile-category cell first."
)
if _missing_cat:
    print(f"⚠️ No _cat column for {_missing_cat} -- these traits are skipped in the sweep.")


def _combo_tag_counts(cell_type, trait, level):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{trait}__{level}__{cell_type}__n{N_OUTER_SPLITS_TERTILE_COUNTS}")


def _write_csv_atomic_counts(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, sep=';', decimal=',')
    os.replace(tmp, path)   # never leaves a half-written file if the kernel dies mid-write


def _write_excel_counts(summary_rows, results, crash_rows, path):
    """Rebuild the whole workbook from memory; written to a temp file, then swapped in."""
    first = ["cell_type", "trait", "tertile"]
    splits, genes = [], []
    for (ct, trait, lv), r in results.items():
        if r.get("df_splits") is not None:
            splits.append(r["df_splits"].assign(cell_type=ct, trait=trait, tertile=lv))
        g = r.get("genes_across_splits")
        if g is not None and len(g):
            g = g.reset_index()
            g = g.rename(columns={g.columns[0]: "Gene"})
            genes.append(g.assign(cell_type=ct, trait=trait, tertile=lv))
    df_sum = pd.DataFrame(summary_rows)
    if len(df_sum):
        df_sum = df_sum.sort_values("test_auc_mean", ascending=False)
    df_spl = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=first)
    df_gen = pd.concat(genes, ignore_index=True) if genes else pd.DataFrame(columns=first)
    df_spl = df_spl[first + [c for c in df_spl.columns if c not in first]]
    df_gen = df_gen[first + [c for c in df_gen.columns if c not in first]]
    settings = pd.DataFrame({
        "setting": ["biomarker", "features", "n_outer_splits", "n_iterations", "quantile_low", "quantile_high",
                    "cell_types", "traits"],
        "value": [BIOMARKER, "mean CP10k of raw counts per donor (counts / total_counts * 1e4), no scVI",
                  N_OUTER_SPLITS_TERTILE_COUNTS, N_ITER_PDF, quantile_low, quantile_high,
                  ", ".join(map(str, CELL_TYPES_FOR_TERTILE_COUNTS)), ", ".join(TRAITS_FOR_TERTILE)]})
    tmp = path[:-5] + ".tmp.xlsx"   # keep the .xlsx suffix so pandas picks the openpyxl engine
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        df_sum.to_excel(xw, sheet_name="summary", index=False)
        df_spl.to_excel(xw, sheet_name="splits", index=False)
        df_gen.to_excel(xw, sheet_name="genes", index=False)
        pd.DataFrame(crash_rows, columns=["cell_type", "trait", "tertile", "error"]).to_excel(
            xw, sheet_name="crashes", index=False)
        settings.to_excel(xw, sheet_name="settings", index=False)
    os.replace(tmp, path)


# --- resume: load what previous runs already saved (same number of splits only; other rows are kept) ---
all_rows_tertile_counts = []
_other_rows_counts = pd.DataFrame()
per_celltype_results_tertile_counts = {}   # (cell_type, trait, level) -> {"df_splits", "summary", "genes_across_splits"}
done_tertile_counts = set()
if RESUME_TERTILE_COUNTS and os.path.exists(SUMMARY_TERTILE_COUNTS_CSV):
    _prev = pd.read_csv(SUMMARY_TERTILE_COUNTS_CSV, sep=';', decimal=',')
    _same = _prev["n_splits_requested"] == N_OUTER_SPLITS_TERTILE_COUNTS
    _other_rows_counts, _prev = _prev[~_same], _prev[_same]
    all_rows_tertile_counts = _prev.to_dict("records")
    for r in all_rows_tertile_counts:
        key = (r["cell_type"], r["trait"], r["tertile"])
        done_tertile_counts.add(key)
        tag = _combo_tag_counts(*key)
        _p_splits = f"{COMBO_DIR_TERTILE_COUNTS}/{tag}__splits.csv"
        _p_genes = f"{COMBO_DIR_TERTILE_COUNTS}/{tag}__genes.csv"
        per_celltype_results_tertile_counts[key] = {
            "summary": r,
            "df_splits": pd.read_csv(_p_splits, sep=';', decimal=',') if os.path.exists(_p_splits) else None,
            "genes_across_splits": (pd.read_csv(_p_genes, sep=';', decimal=',', index_col=0)
                                    if os.path.exists(_p_genes) else None),
        }
    print(f"↩️ Resuming: {len(done_tertile_counts)} combos already done (loaded from {SUMMARY_TERTILE_COUNTS_CSV})")

crash_rows_tertile_counts = []
_todo = [(t, ct, lv) for t in TRAITS_FOR_TERTILE if f"{t}_cat" in df_pb_latent.columns
         for ct in CELL_TYPES_FOR_TERTILE_COUNTS for lv in TERTILE_LEVELS
         if (ct, t, lv) not in done_tertile_counts]
print(f"▶️ {len(_todo)} combos to run (features: normalized counts, no scVI) -> {os.path.abspath(OUTDIR_TERTILE_COUNTS)}")

for i_combo, (trait, cell_type, level) in enumerate(_todo, 1):
    cat_col = f"{trait}_cat"
    print(f"\n===== [{i_combo}/{len(_todo)}] {cell_type} | {trait} = {level} (counts) =====")
    try:
        res = run_discovery_multi_split(
            model=COUNTS_MODEL, adata=adata, df_clinical=df_pb_latent,
            biomarker=BIOMARKER, cell_type=cell_type,
            n_iterations=N_ITER_PDF, target_group=level,
            group_col=cat_col,
            n_outer_splits=N_OUTER_SPLITS_TERTILE_COUNTS,
        )
    except Exception as e:
        print(f"⚠️ CRASH on {cell_type} ({trait}={level}): {e}")
        crash_rows_tertile_counts.append({"cell_type": cell_type, "trait": trait, "tertile": level,
                                          "error": repr(e)})
        _write_csv_atomic_counts(pd.DataFrame(crash_rows_tertile_counts), CRASH_TERTILE_COUNTS_CSV)
        continue

    key = (cell_type, trait, level)
    per_celltype_results_tertile_counts[key] = res
    s = res["summary"]
    all_rows_tertile_counts.append({
        "cell_type": cell_type, "trait": trait, "tertile": level,
        "n_splits_valid": s["n_splits_valid"], "n_splits_requested": s["n_splits_requested"],
        "n_test_mean": res["df_splits"]["n_test"].mean(),
        "test_auc_mean": s["test_auc_mean"], "test_auc_std": s["test_auc_std"],
        "inner_val_auc_mean": s["inner_val_auc_mean"], "inner_val_auc_std": s["inner_val_auc_std"],
        "train_auc_mean": s["train_auc_mean"],
        "overfit_gap_mean": s["overfit_gap_mean"], "overfit_gap_std": s["overfit_gap_std"],
        "overfit_gap_innerval_mean": s["overfit_gap_innerval_mean"],
        "overfit_gap_innerval_std": s["overfit_gap_innerval_std"],
        "features": "counts",
    })

    # --- progressive save: this combo's details + the full summary so far (CSV, then Excel) ---
    tag = _combo_tag_counts(*key)
    _write_csv_atomic_counts(res["df_splits"], f"{COMBO_DIR_TERTILE_COUNTS}/{tag}__splits.csv")
    _g = res["genes_across_splits"].reset_index()
    _g = _g.rename(columns={_g.columns[0]: "Gene"})
    _write_csv_atomic_counts(_g, f"{COMBO_DIR_TERTILE_COUNTS}/{tag}__genes.csv")
    _write_csv_atomic_counts(pd.concat([pd.DataFrame(all_rows_tertile_counts), _other_rows_counts], ignore_index=True)
                             .sort_values("test_auc_mean", ascending=False), SUMMARY_TERTILE_COUNTS_CSV)
    try:
        _write_excel_counts(all_rows_tertile_counts, per_celltype_results_tertile_counts, crash_rows_tertile_counts,
                            SUMMARY_TERTILE_COUNTS_XLSX)
    except Exception as e:   # the CSVs are the resume source; an Excel problem must not stop the sweep
        print(f"⚠️ Excel write failed ({e!r}); CSVs are saved, the workbook will be rebuilt after the next combo.")
    print(f"💾 saved ({len(all_rows_tertile_counts)} combos in {SUMMARY_TERTILE_COUNTS_CSV} + .xlsx)")
    gc.collect()

_write_excel_counts(all_rows_tertile_counts, per_celltype_results_tertile_counts, crash_rows_tertile_counts,
                    SUMMARY_TERTILE_COUNTS_XLSX)
df_all_celltypes_tertile_counts = pd.DataFrame(all_rows_tertile_counts)
if len(df_all_celltypes_tertile_counts):
    df_all_celltypes_tertile_counts = df_all_celltypes_tertile_counts.sort_values("test_auc_mean", ascending=False)
print(f"📗 Excel: {os.path.abspath(SUMMARY_TERTILE_COUNTS_XLSX)}")
if crash_rows_tertile_counts:
    print(f"⚠️ {len(crash_rows_tertile_counts)} combos crashed -- see {CRASH_TERTILE_COUNTS_CSV}; "
          f"re-run to retry them.")
