"""Two-level estimators (all inner steps use only the training donors they are given)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
import evalkit as ek

ALPHAS = np.logspace(-1, 6, 30)


def _multi_ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=ALPHAS, alpha_per_target=True))


def _inner_oof(make, X, Y, n_splits=5, seed=0, w=None):
    """Out-of-fold predictions of a (multi-output) model inside the training set, plus the model refit on all.
    w: optional sample weights for the last pipeline step."""
    Y = np.asarray(Y, dtype=float)
    Y2 = Y if Y.ndim == 2 else Y[:, None]
    oof = np.zeros_like(Y2)
    fp = lambda idx: {} if w is None else {"ridgecv__sample_weight": w[idx]}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        m = make().fit(X.iloc[tr], Y2[tr], **fp(tr))
        p = m.predict(X.iloc[te])
        oof[te] = p if p.ndim == 2 else p[:, None]
    full = make().fit(X, Y2, **fp(np.arange(len(X))))
    return oof, full


class ProxyModel(BaseEstimator, RegressorMixin):
    """snRNA-seq -> predicted age-26 traits (+ predicted pace) -> pace.
    Clinical traits are used only as TRAINING labels; at prediction time only snRNA-seq is needed.
    aux: trait columns of the donor table. keep_direct: include a direct snRNA->pace prediction.
    extra_cols: columns of X passed straight to the second level (e.g. measured clinical)."""

    def __init__(self, aux=(), keep_direct=True, snrna_prefix="pb|", extra_prefix=None, weight=None, seed=0):
        self.aux, self.keep_direct, self.snrna_prefix, self.extra_prefix = aux, keep_direct, snrna_prefix, extra_prefix
        self.weight, self.seed = weight, seed

    def _split(self, X):
        sn = [c for c in X.columns if c.startswith(self.snrna_prefix) or c.startswith("cov|")]
        ex = [c for c in X.columns if self.extra_prefix and c.startswith(self.extra_prefix)]
        return X[sn], X[ex]

    def _labels(self, X, y):
        d = ek.load().join(ek.aux_future(), how="left").reindex(X.index)
        L = d[list(self.aux)].apply(pd.to_numeric, errors="coerce")
        L = L.fillna(L.median())
        if self.keep_direct:
            L = L.assign(__pace__=y)
        return L.to_numpy(dtype=float)

    def fit(self, X, y):
        Xs, Xe = self._split(X)
        L = self._labels(X, np.asarray(y, float))
        self.mu_, self.sd_ = L.mean(0), L.std(0) + 1e-9
        w = None
        if self.weight == "sqrt":
            w = np.sqrt(ek.load().reindex(X.index)["n_cells"].to_numpy(float))
        oof, self.first_ = _inner_oof(_multi_ridge, Xs, (L - self.mu_) / self.sd_, seed=self.seed, w=w)
        Z = np.hstack([oof, Xe.to_numpy(dtype=float)]) if Xe.shape[1] else oof
        self.second_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     RidgeCV(alphas=np.logspace(-3, 4, 30))).fit(Z, y)
        return self

    def predict(self, X):
        Xs, Xe = self._split(X)
        Z = self.first_.predict(Xs)
        Z = Z if Z.ndim == 2 else Z[:, None]
        Z = np.hstack([Z, Xe.to_numpy(dtype=float)]) if Xe.shape[1] else Z
        return self.second_.predict(Z)


class ViewEnsemble(BaseEstimator, RegressorMixin):
    """One ridge per view (column-prefix group, each view also gets the cov| columns);
    combine = 'mean' (average of standardized predictions) or 'stack' (ridge on inner out-of-fold predictions)."""

    def __init__(self, views=(), combine="stack", extra_prefix=None):
        self.views, self.combine, self.extra_prefix = views, combine, extra_prefix

    def _cols(self, X, v):
        return [c for c in X.columns if c.startswith(v) or c.startswith("cov|")]

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.models_, cols_oof = [], []
        for v in self.views:
            Xv = X[self._cols(X, v)]
            oof, full = _inner_oof(_multi_ridge, Xv, y)
            self.models_.append(full)
            cols_oof.append(oof[:, 0])
        Z = np.column_stack(cols_oof)
        self.zmu_, self.zsd_ = Z.mean(0), Z.std(0) + 1e-9
        ex = [c for c in X.columns if self.extra_prefix and c.startswith(self.extra_prefix)]
        self.ex_ = ex
        if self.combine == "stack" or ex:
            Zf = np.hstack([Z, X[ex].to_numpy(float)]) if ex else Z
            self.second_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         RidgeCV(alphas=np.logspace(-3, 4, 30))).fit(Zf, y)
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(X[self._cols(X, v)]).ravel() for m, v in zip(self.models_, self.views)])
        if self.combine == "mean" and not self.ex_:
            return ((Z - self.zmu_) / self.zsd_).mean(1)
        Zf = np.hstack([Z, X[self.ex_].to_numpy(float)]) if self.ex_ else Z
        return self.second_.predict(Zf)


class Weighted(BaseEstimator, RegressorMixin):
    """Wrap an sklearn pipeline so training donors are weighted by a function of their cell count
    (sample_weight goes to the last step). mode: 'sqrt' | 'log' | 'min100' (drop <100 cells from training)."""

    def __init__(self, make=None, mode="sqrt"):
        self.make, self.mode = make, mode

    def _w(self, idx):
        n = ek.load().reindex(idx)["n_cells"].to_numpy(float)
        if self.mode == "sqrt":
            return np.sqrt(n)
        if self.mode == "log":
            return np.log1p(n)
        if self.mode == "min100":
            return (n >= 100).astype(float)
        if self.mode == "cap":
            return np.minimum(n, 300.0)
        raise ValueError(self.mode)

    def fit(self, X, y):
        w = self._w(X.index)
        keep = w > 0
        self.m_ = self.make()
        last = self.m_.steps[-1][0]
        self.m_.fit(X[keep], np.asarray(y)[keep], **{f"{last}__sample_weight": w[keep]})
        return self

    def predict(self, X):
        return self.m_.predict(X)


class Bagged(BaseEstimator, RegressorMixin):
    """Average of several copies of an estimator that differ only in their inner-CV seed."""

    def __init__(self, make=None, n=5):
        self.make, self.n = make, n

    def fit(self, X, y):
        self.ms_ = [self.make(seed).fit(X, y) for seed in range(self.n)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms_], axis=0)


class BySex(BaseEstimator, RegressorMixin):
    """Separate model per sex (uses the cov|sex_M column); predictions z-scored within sex on the training fit."""

    def __init__(self, make=None):
        self.make = make

    def fit(self, X, y):
        y = np.asarray(y, float)
        s = X["cov|sex_M"].to_numpy()
        self.ms_ = {}
        for v in (0.0, 1.0):
            m = s == v
            self.ms_[v] = self.make().fit(X[m], y[m])
        return self

    def predict(self, X):
        s = X["cov|sex_M"].to_numpy()
        out = np.zeros(len(X))
        for v, m in self.ms_.items():
            k = s == v
            if k.any():
                out[k] = m.predict(X[k])
        return out
