"""Full 5x3 CV (same folds as the overnight search) for the MIL models. High = top 10%, Low = bottom 60% (v11).
Usage: python run_mil.py MODEL [MODEL ...]   with MODEL in meanpool, abmil, mixmil, scrat, cloudpred, scmild"""
import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, torch, evalkit as ek, mil_models as mm
ek.Q_LOW, ek.Q_HIGH = 0.6, 0.9
torch.set_num_threads(4)
ids = ek.dev_donors(); mm.load_bags(ids)
X = pd.DataFrame({"x": 0}, index=ids)
MODELS = {
    "meanpool":  ("MIL00 mean-pooled scVI latent (+per-CT means) | logit L2", lambda: mm.MeanPoolLogit()),
    "abmil":     ("MIL01 ABMIL gated attention on scVI latent", lambda: mm.ABMIL()),
    "mixmil":    ("MIL02 MixMIL (official) on scVI latent", lambda: mm.MixMILModel()),
    "scrat":     ("MIL03 ScRAT (official Transformer + mixup) on scVI latent",
                  lambda: mm.ScRATModel(bags_per_donor=10, augment_num=200, epochs=40)),
    "cloudpred": ("MIL04 CloudPred (official) on scVI latent", lambda: mm.CloudPredModel(centers=(5, 10))),
    "scmild":    ("MIL05 scMILD (official modules) on scVI latent", lambda: mm.ScMILDModel()),
}
for m in sys.argv[1:]:
    name, fac = MODELS[m]
    ek.evaluate(name, X, fac, task="clf", notes="MIL_PaceOfAging, q60-90, scVI latent 30d")
