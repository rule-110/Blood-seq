import sys; sys.path.insert(0, "/home/nlapique/Blood/script/NIGHT_PaceOfAging_snRNA/scripts")
import pandas as pd, numpy as np
from scipy import stats
import evalkit as ek
from common_models import *
from meta_models import Weighted
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from xgboost import XGBRegressor

ids = ek.dev_donors(); cov = ek.covariates(ids)
pb3 = ek.pseudobulk(ids, ["ALL"], min_cpm=3)
X = pd.concat([cov, pb3], axis=1)
y = ek.load().reindex(ids)[ek.TARGET]

# rank-transformed target (robust to outlying pace values)
yr = pd.Series(stats.rankdata(y) / len(y), index=ids)
ek.evaluate("R01 pbALL3 +cov | ridge on rank(pace)", X, ridge(), y=yr, notes="metrics vs rank(pace), AUC unaffected")
# quantile-normal genes (fit inside fold)
qn = lambda: make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                           StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 30)))
ek.evaluate("R02 pbALL3 +cov | quantile-normal genes ridge", X, qn)
# classifier on extremes
ek.evaluate("R03 pbALL3 +cov | logitL2 on extremes", X, logit_l2(), task="clf")
ek.evaluate("R04 pbALL3 +cov | logitL2 top20 vs rest", X, logit_l2(), task="clf_top")
# kernel ridge rbf on PCA
kr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(100),
                           GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [0.1, 1, 10], "gamma": [1e-4, 1e-3, 1e-2]}, cv=5))
ek.evaluate("R05 pbALL3 +cov | pca100 kernel-ridge rbf", X, kr)
# SVR linear-ish on screened genes
sv = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_regression, k=1000),
                           GridSearchCV(SVR(kernel="linear"), {"C": [1e-4, 1e-3, 1e-2]}, cv=5))
ek.evaluate("R06 pbALL3 +cov | screen1000 linear SVR", X, sv)
# XGBoost on screened genes
xg = lambda: make_pipeline(SimpleImputer(strategy="median"), SelectKBest(f_regression, k=300),
                           XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.02, subsample=0.7,
                                        colsample_bytree=0.3, min_child_weight=10, reg_lambda=10, n_jobs=8))
ek.evaluate("R07 pbALL3 +cov | screen300 xgboost d2", X, xg)
# screening depth for ridge
for k in (2000, 5000):
    ek.evaluate(f"R08 pbALL3 +cov | screen{k} ridge", X, screen_ridge(k))
for n in (2, 5, 10):
    ek.evaluate(f"R09 pbALL3 +cov | pls{n}", X, pls(n))
