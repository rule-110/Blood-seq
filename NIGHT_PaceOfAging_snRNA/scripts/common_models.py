"""Model factories used across experiments."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


def ridge(alphas=np.logspace(-1, 6, 30)):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=alphas))


def enet(l1=(0.1, 0.5, 0.9), n_alphas=30):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 ElasticNetCV(l1_ratio=list(l1), alphas=n_alphas, cv=5, max_iter=20000, n_jobs=8))


def screen_ridge(k):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 SelectKBest(f_regression, k=k), RidgeCV(alphas=np.logspace(-1, 6, 30)))


def screen_enet(k):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 SelectKBest(f_regression, k=k),
                                 ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=30, cv=5, max_iter=20000, n_jobs=8))


def pca_ridge(n):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=n),
                                 RidgeCV(alphas=np.logspace(-2, 5, 30)))


class _PLS(PLSRegression):
    def predict(self, X):
        return super().predict(X).ravel()


def pls(n):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _PLS(n_components=n, scale=False))


def logit_l2():
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=5, scoring="roc_auc",
                                                      max_iter=5000, n_jobs=8))
