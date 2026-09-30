"""sc-ImmuAging clock scores as predictors of PaceOfAgingP45 (dev donors only, same folds as exp01-13).
Clock scores come from prep5_scimmuaging.py. Nothing is fitted on our donors to build the clocks."""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import evalkit as ek
from common_models import ridge
from meta_models import ProxyModel

CLOCKS = ["CD4T", "CD8T", "MONO", "NK", "B"]
MIN_CELLS = 20
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]

ids = ek.dev_donors()
d = ek.load().reindex(ids)
gi = pd.read_pickle(ek.DATA + "scimmuaging_groups.pkl")
gi = gi[gi.patient_id.isin(set(ek.load().index))]


def donor_scores(gi, libcorr):
    g = gi.copy()
    if libcorr:   # subtract each library's mean score (dev donors, cell-weighted), like pseudobulk_libcorr
        dev = g.patient_id.isin(set(ids))
        for c in CLOCKS:
            r = g.clock_ct == c
            grand = np.average(g.loc[r & dev, "score"], weights=g.loc[r & dev, "n_cells"])
            for L, sub in g[r & dev].groupby("library"):
                if len(sub) >= 3:
                    g.loc[r & (g.library == L), "score"] -= np.average(sub.score, weights=sub.n_cells) - grand
    g["ws"] = g.score * g.n_cells
    agg = g.groupby(["patient_id", "clock_ct"])[["ws", "n_cells"]].sum()
    sc = (agg.ws / agg.n_cells).where(agg.n_cells >= MIN_CELLS).unstack()[CLOCKS]
    return sc.reindex(ids)


raw = donor_scores(gi, libcorr=False)
lc = donor_scores(gi, libcorr=True)

# ---------------- sanity checks (descriptive, dev donors) ----------------
print("donors with a score (>= %d cells):" % MIN_CELLS, raw.notna().sum().to_dict())
print("\nmean predicted age (raw):", raw.mean().round(1).to_dict())
print("\nclock-clock Spearman (library-corrected):\n", lc.corr("spearman").round(2).to_string())
S = d[ek.SMOKING].fillna(d[ek.SMOKING].median())
pace_res = d[ek.TARGET] - LinearRegression().fit(S, d[ek.TARGET]).predict(S)
ref = pd.DataFrame({
    "sex_M": d.sex_M, "CurrentSmk26": d.CurrentSmk26, "PackYrsBy26": d.PackYrsBy26, "LnCRP26": d.LnCRP26,
    "bmi26": d.bmi26, "VOmax26": d.VOmax26, "CD8naive26(DNAm)": d.CD8naive26, "CD4naive26(DNAm)": d.CD4naive26,
    "log_ncells": np.log(d.n_cells), "mean_umi": d.mean_umi, "pct_mt": d.mean_pct_mt,
    "PACE": d[ek.TARGET], "PACE_resid_smoking": pace_res})
rows = []
for name, sc in [("raw", raw), ("libcorr", lc)]:
    for c in CLOCKS:
        for v in ref.columns:
            ok = sc[c].notna() & ref[v].notna()
            r, p = stats.spearmanr(sc.loc[ok, c], ref.loc[ok, v])
            rows.append({"scores": name, "clock": c, "vs": v, "rho": r, "p": p, "n": int(ok.sum())})
cor = pd.DataFrame(rows)
cor.to_csv(ek.RES + "scimmuaging_correlations.csv", index=False)
piv = cor.pivot_table(index=["scores", "vs"], columns="clock", values="rho")[CLOCKS]
print("\nSpearman rho of clock score with:\n", piv.round(2).to_string())
pv = cor.pivot_table(index=["scores", "vs"], columns="clock", values="p")[CLOCKS]
print("\np-values for PACE rows:\n", pv.loc[(slice(None), ["PACE", "PACE_resid_smoking"]), :].applymap(lambda x: f"{x:.2g}").to_string())

# ---------------- CV (same folds as all previous experiments) ----------------
cov = ek.covariates(ids)
Lpb = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3); Lpb.columns = [c.replace("pbL|", "pb|") for c in Lpb.columns]
smk = ek.clinical(ids, ek.SMOKING)
clk = lc.add_prefix("clk|")
clk_raw = raw.add_prefix("clk|")
res = {}
res["C01"] = ek.evaluate("C01 sc-ImmuAging 5 clocks (libcorr) + cov | ridge", pd.concat([cov, clk], axis=1), ridge())
res["C02"] = ek.evaluate("C02 sc-ImmuAging 5 clocks (raw) + cov | ridge", pd.concat([cov, clk_raw], axis=1), ridge())
res["C03"] = ek.evaluate("C03 proxy Y02 + clocks at 2nd level", pd.concat([cov, Lpb, clk], axis=1),
                         lambda: ProxyModel(aux=ek.CLIN13 + BLOOD, extra_prefix="clk|"))
res["C04"] = ek.evaluate("C04 smoking + clocks + cov | ridge", pd.concat([cov, smk, clk], axis=1), ridge())
res["C05"] = ek.evaluate("C05 clocks + cov -> pace residualized on smoking", pd.concat([cov, clk], axis=1), ridge(),
                         y=pace_res)
res["C06"] = ek.evaluate("C06 cov only -> pace residualized on smoking", cov, ridge(), y=pace_res)


# ---------------- paired comparisons (Nadeau-Bengio corrected resampled t-test on per-fold AUCs) ----------------
def fold_aucs(name):
    return np.load(ek.RES + f"oof/{ek.safe(name)}.npz")["aucs"]


def paired(a, b, label):
    x, y = fold_aucs(a), fold_aucs(b)
    diff = x - y; k = len(diff); n_te_over_tr = 1 / (ek.N_FOLDS - 1)
    se = np.sqrt((1 / k + n_te_over_tr) * np.var(diff, ddof=1))
    t = diff.mean() / se; p = 2 * stats.t.sf(abs(t), k - 1)
    print(f"{label:55s} dAUC {diff.mean():+.3f}  corrected p = {p:.3f}")


print()
paired("C01 sc-ImmuAging 5 clocks (libcorr) + cov | ridge", "B00 covariates | ridge", "clocks vs covariates only")
paired("C03 proxy Y02 + clocks at 2nd level", "Y02 proxy[clin13+blood5]+direct libcorr3", "proxy + clocks vs proxy (Y02)")
paired("C04 smoking + clocks + cov | ridge", "B07 smoking only+cov | ridge", "smoking + clocks vs smoking only")
paired("C05 clocks + cov -> pace residualized on smoking", "C06 cov only -> pace residualized on smoking",
       "clocks vs cov, smoking-residualized pace")
