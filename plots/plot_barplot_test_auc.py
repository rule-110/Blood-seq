import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = "/home/nlapique/Blood/script/MULTISPLIT_PaceOfAgingP45_old/"
df = pd.read_csv(d + "summary_all_celltypes_by_sex.csv", sep=";", decimal=",")
order = df.groupby("cell_type")["test_auc_mean"].mean().sort_values(ascending=False).index.tolist()
sexes = [("M", "Male", "#2a78d6"), ("F", "Female", "#eb6834")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(order)); w = 0.36
for i, (s, lab, c) in enumerate(sexes):
    sub = df[df.sex == s].set_index("cell_type").loc[order]
    xx = x + (i - 0.5) * (w + 0.04)
    ax.bar(xx, sub.test_auc_mean - 0.5, bottom=0.5, width=w, color=c, label=lab, zorder=2)
    ax.errorbar(xx, sub.test_auc_mean, yerr=sub.test_auc_std, fmt="none",
                ecolor="#52514e", elinewidth=1, capsize=2.5, zorder=3)
    for v, sd, xv in zip(sub.test_auc_mean, sub.test_auc_std, xx):
        ax.text(xv, max(v + sd, 0.5) + 0.008, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#0b0b0b")

ax.axhline(0.5, color="#52514e", lw=1.2, zorder=4)
ax.text(-0.88, 0.497, "chance (AUC = 0.5)", ha="left", va="top",
        fontsize=8.5, color="#52514e")
ax.set_xticks(x); ax.set_xticklabels([o.replace("Non-classical ", "Non-classical\n")
                                      .replace("Classical ", "Classical\n") for o in order])
ax.set_ylim(0.3, 0.82); ax.set_xlim(-0.9, len(order) - 0.4)
ax.set_ylabel("Test AUC (mean ± SD over 5 splits)", color="#52514e")
ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
for sp in ["top", "right", "bottom"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#bdbcb6"); ax.tick_params(colors="#52514e", length=0)
ax.set_title("PaceOfAgingP45 classification: test AUC by cell type and sex",
             loc="left", fontsize=11.5, fontweight="bold", color="#0b0b0b", pad=24)
ax.text(0, 1.02, "Bars start at chance (0.5). Sorted by mean AUC across sexes. Train AUC = 1.0 for every model.",
        transform=ax.transAxes, fontsize=8.5, color="#52514e")
ax.legend(loc="upper right", bbox_to_anchor=(1, 1.1), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(d + f"barplot_test_auc_by_celltype_sex.{ext}", dpi=200)
