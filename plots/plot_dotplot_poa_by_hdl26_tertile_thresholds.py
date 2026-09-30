import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Same label rule as run_discovery_pdf_report (label_cutoffs="group"):
# within each tertile, High = pace >= 90th pct, Low = pace <= 70th pct, in between = not used.
Q_LOW, Q_HIGH = 0.7, 0.9
out = "/home/nlapique/Blood/script/MULTISPLIT_PaceOfAgingP45_old/"
df = pd.read_csv("/home/nlapique/Downloads/BroadInst_DunedinPheno_20240829.txt", sep="\t")
df = df[["snum", "hdl26", "PaceOfAgingP45"]].dropna()
df["tertile"], edges = pd.qcut(df.hdl26, 3, labels=["T1 (low)", "T2", "T3 (high)"], retbins=True)
g_lo, g_hi = df.PaceOfAgingP45.quantile([Q_LOW, Q_HIGH])

C_HIGH, C_LOW, C_MID = "#eb6834", "#2a78d6", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9, 6.5))
rng = np.random.default_rng(0)
labels = []
for i, (t, g) in enumerate(df.groupby("tertile", observed=True)):
    v = g.PaceOfAgingP45.values
    lo, hi = np.quantile(v, [Q_LOW, Q_HIGH])
    cls = np.where(v >= hi, "H", np.where(v <= lo, "L", "M"))
    x = i + rng.uniform(-0.24, 0.24, len(v))
    for c, col, z in [("M", C_MID, 2), ("L", C_LOW, 3), ("H", C_HIGH, 4)]:
        k = cls == c
        ax.scatter(x[k], v[k], s=15, color=col, alpha=0.85 if c != "M" else 0.7,
                   edgecolor="white", linewidth=0.4, zorder=z)
    for y, ls, name in [(hi, "-", "≥"), (lo, (0, (4, 2)), "≤")]:
        ax.hlines(y, i - 0.3, i + 0.3, color="#0b0b0b", lw=1.6, ls=ls, zorder=5)
        ax.text(i + 0.32, y, f"{name} {y:.2f}", va="center", fontsize=8.5, color="#0b0b0b")
    nH, nL = (cls == "H").sum(), (cls == "L").sum()
    labels.append(f"{t}\nHDL {edges[i]:.2f}–{edges[i+1]:.2f}\nn = {len(v)}\n{nH} High / {nL} Low")

ax.axhline(g_hi, color="#52514e", lw=1, ls=":", zorder=1)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_xlim(-0.4, 2.6)
ax.set_ylabel("PaceOfAgingP45", color="#52514e")
ax.set_xlabel("HDL at age 26, tertiles (mmol/L)", color="#52514e")
ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
for sp in ["top", "right", "bottom"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#bdbcb6"); ax.tick_params(colors="#52514e", length=0)
ax.set_title("Pace-of-aging label cutoffs within each HDL tertile", loc="left",
             fontsize=11.5, fontweight="bold", color="#0b0b0b", pad=30)
ax.text(0, 1.035, "Cutoffs = 70th / 90th percentile of pace inside each tertile, as in the classifier "
        "(label_cutoffs=\"group\")", transform=ax.transAxes, fontsize=8.5, color="#52514e")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", markersize=7, label=l)
                   for c, l in [(C_HIGH, "High (≥ 90th pct)"), (C_MID, "Not used"), (C_LOW, "Low (≤ 70th pct)")]]
          + [Line2D([], [], color="#52514e", lw=1, ls=":", label=f"Whole-cohort 90th pct ({g_hi:.2f})")],
          loc="upper right", bbox_to_anchor=(1, 1.03), frameon=False, fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(out + f"dotplot_PaceOfAgingP45_by_hdl26_tertile_thresholds.{ext}", dpi=200)
