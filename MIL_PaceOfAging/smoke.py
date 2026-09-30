import sys, time; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import numpy as np, pandas as pd, evalkit as ek, mil_models as mm
from sklearn.metrics import roc_auc_score
ids = ek.dev_donors(); mm.load_bags(ids)
d = ek.load().reindex(ids); y = d[ek.TARGET].to_numpy()
tr, te = ek.splits(len(ids))[0]
lo, hi = np.quantile(y[tr], .6), np.quantile(y[tr], .9)
use = tr[(y[tr] <= lo) | (y[tr] >= hi)]; X = pd.DataFrame(index=ids, data={"x": 0})
ext = te[(y[te] <= lo) | (y[te] >= hi)]
for name, m in [("mixmil", mm.MixMILModel(n_epochs=100)),
                ("cloudpred", mm.CloudPredModel(centers=(5,)))]:
    t0 = time.time()
    try:
        m.fit(X.iloc[use], (y[use] >= hi).astype(int))
        p = m.predict_proba(X.iloc[ext])[:, 1]
        print(f"{name:10s} AUC {roc_auc_score((y[ext] >= hi).astype(int), p):.3f}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(name, "FAILED", repr(e)[:300], flush=True)
