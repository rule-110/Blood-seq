import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
import evalkit as ek
from common_models import ridge
from meta_models import ProxyModel
ids = ek.dev_donors(); cov = ek.covariates(ids)
L = ek.pseudobulk_libcorr(ids, "ALL", min_cpm=3); L.columns = [c.replace("pbL|", "pb|") for c in L.columns]
smk = ek.clinical(ids, ek.SMOKING)
BLOOD = ["neutro26np", "lymph26np", "mono26np", "eosino26np", "baso26np"]
ek.evaluate("Z01 smoking + snRNA proxy (smoking measured)", pd.concat([cov, L, smk], axis=1), lambda: ProxyModel(aux=ek.CLIN13 + BLOOD, extra_prefix="clin|"))
d = ek.load().reindex(ids)
never = (d["CurrentSmk26"] == 0) & (d["PackYrsBy26"].fillna(0) == 0)
print(f"never-smokers among dev: {never.sum()}", flush=True)
# pace residualized on smoking (linear, dev-wide; label transform only)
from sklearn.linear_model import LinearRegression
S = d[ek.SMOKING].fillna(d[ek.SMOKING].median())
res = d[ek.TARGET] - LinearRegression().fit(S, d[ek.TARGET]).predict(S)
ek.evaluate("Z02 snRNA proxy -> pace residualized on smoking", pd.concat([cov, L], axis=1), lambda: ProxyModel(aux=ek.CLIN13 + BLOOD), y=res)
ek.evaluate("Z03 snRNA ridge -> pace residualized on smoking", pd.concat([cov, L], axis=1), ridge(), y=res)
# top genes of the direct snRNA->pace ridge fit on all dev donors (standardized coefficients)
m = ridge()().fit(pd.concat([cov, L], axis=1), d[ek.TARGET].to_numpy())
coef = pd.Series(m[-1].coef_, index=pd.concat([cov, L], axis=1).columns).drop(cov.columns)
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(40)
top.rename(lambda c: c.split("|")[-1]).to_csv(ek.RES + "top_genes_direct_ridge.csv", header=["std_coef"])
print(top.rename(lambda c: c.split("|")[-1]).round(5).to_string(), flush=True)
