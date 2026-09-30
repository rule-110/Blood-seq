# Multiple-instance learning baselines

Donor = bag of cells (30-d scVI latent). The folds, labels (High = top 10 %, Low = bottom 60 %)
and dev donors are the same as in `NIGHT_PaceOfAging_snRNA` (it imports `evalkit`).

1. Clone the official implementations into `repos/` (git-ignored). `mil_models.py` expects these folder names:

   | folder | source |
   |---|---|
   | `repos/mil_mixmil` | https://github.com/AIH-SGML/mixmil |
   | `repos/mil_ScRAT` | https://github.com/yuzhenmao/ScRAT |
   | `repos/mil_CloudPred` | https://github.com/bryanhe/CloudPred |
   | `repos/mil_scMILD` | official scMILD repository (Jeong et al.). The code imports `src/*.py` |

2. `python prep_latent.py` writes `latent_scvi.npy` and `cells.parquet` here (these are git-ignored).
3. `python smoke.py` / `smoke2.py` for quick checks, then `python run_mil.py meanpool abmil mixmil scrat cloudpred scmild`.

CloudPred took about 3.5 h on CPU.
