import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

out = "/home/nlapique/Blood/script/MULTISPLIT_PaceOfAgingP45_old/"
df = pd.read_csv("/home/nlapique/Downloads/BroadInst_DunedinPheno_20240829.txt", sep="\t")
df = df[["snum", "crp26ns", "PaceOfAgingP45"]].dropna()
df["tertile"] = pd.qcut(df.crp26ns, 3, labels=["T1 (low)", "T2", "T3 (high)"])
edges = pd.qcut(df.crp26ns, 3, retbins=True)[1]
cols = ["#86b6ef", "#3987e5", "#1c5cab"]

rho, p_rho = spearmanr(df.crp26ns, df.PaceOfAgingP45)
H, p_kw = kruskal(*[g.PaceOfAgingP45 for _, g in df.groupby("tertile", observed=True)])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7, 6))
rng = np.random.default_rng(0)
for i, (t, g) in enumerate(df.groupby("tertile", observed=True)):
    v = g.PaceOfAgingP45.values
    ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=14, color=cols[i], alpha=0.6,
               edgecolor="white", linewidth=0.4, zorder=2)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.vlines(i + 0.4, q1, q3, color="#0b0b0b", lw=2, zorder=3)
    ax.hlines(med, i + 0.34, i + 0.46, color="#0b0b0b", lw=2, zorder=3)
    ax.text(i + 0.5, med, f"{med:.2f}", va="center", fontsize=8.5, color="#0b0b0b")

# crp26ns is recorded in whole mg/L, so qcut bins (right-closed) are ties-aware: report the actual ranges
labels = [f"{t}\n{int(g.crp26ns.min())}–{int(g.crp26ns.max())} mg/L\nn = {len(g)}"
          for t, g in df.groupby("tertile", observed=True)]
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_xlim(-0.5, 2.75)
ax.axhline(1.0, color="#52514e", lw=1, ls="--", zorder=1)
ax.set_ylabel("PaceOfAgingP45 (years of aging per year)", color="#52514e")
ax.set_xlabel("CRP at age 26 (crp26ns), tertiles", color="#52514e")
ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
for sp in ["top", "right", "bottom"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#bdbcb6"); ax.tick_params(colors="#52514e", length=0)
ax.set_title("Pace of aging by CRP tertile (age 26)", loc="left", fontsize=11.5,
             fontweight="bold", color="#0b0b0b", pad=24)
ax.text(0, 1.02, f"Black bar = median and IQR.  Spearman ρ = {rho:.2f} (p = {p_rho:.1e});  "
        f"Kruskal–Wallis p = {p_kw:.1e}", transform=ax.transAxes, fontsize=8.5, color="#52514e")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(out + f"dotplot_PaceOfAgingP45_by_crp26ns_tertile.{ext}", dpi=200)
print(len(df), edges, rho, p_rho, p_kw)
print(df.groupby("tertile", observed=True).PaceOfAgingP45.describe())
