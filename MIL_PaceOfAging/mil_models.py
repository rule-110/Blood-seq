"""Cell-level (multiple-instance) classifiers for PaceOfAgingP45, all fed the same scVI latent (30 dims per cell).

Each class is a sklearn-style estimator whose X is a DataFrame indexed by donor id (the columns are ignored):
the donor's cells are looked up in BAGS. evalkit.evaluate(task="clf") gives the same 5x3 folds as every other
experiment and fits on the training fold's extreme donors only.

  MeanPoolLogit : mean latent per donor (+ per-cell-type means) -> L2 logistic regression (non-MIL baseline)
  ABMIL         : gated-attention MIL (Ilse et al. 2018)
  MixMILModel   : MixMIL (Engelmann et al., AISTATS 2024), official package, tensor code path
  ScRATModel    : ScRAT (Mao et al., Bioinformatics 2024): official TransformerPredictor + cell-type-stratified
                  patient mixup + bag sampling (500 cells) + multi-bag averaging at test time
  CloudPredModel: CloudPred (He et al., PSB 2022): official Gaussian-mixture DensityClassifier + train()
"""
import sys, types, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

REPOS = "/home/nlapique/Blood/script/MIL_PaceOfAging/repos/"   # official code, cloned 2026-09-29
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAGS, CTS, CT_LIST = {}, {}, []


def load_bags(ids):
    Z = np.load("/home/nlapique/Blood/script/MIL_PaceOfAging/latent_scvi.npy")
    cells = pd.read_parquet("/home/nlapique/Blood/script/MIL_PaceOfAging/cells.parquet")
    CT_LIST[:] = sorted(cells.CellType.unique())
    ctcode = cells.CellType.map({c: i for i, c in enumerate(CT_LIST)}).to_numpy()
    ids = set(ids)
    for pid, idx in cells.groupby("patient_id").indices.items():
        if pid in ids:
            BAGS[pid] = Z[idx]; CTS[pid] = ctcode[idx]
    return BAGS


def _sample(pid, n, rng):
    b = BAGS[pid]
    return b[rng.choice(len(b), n, replace=len(b) < n)]


def _inner_split(ids, y, seed):
    return train_test_split(np.asarray(ids), np.asarray(y), test_size=0.15, random_state=seed, stratify=y)


class _Base(BaseEstimator, ClassifierMixin):
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = self._score(list(X.index))
        return np.column_stack([1 - p, p])


# ------------------------------------------------------------------ baseline
class MeanPoolLogit(_Base):
    def __init__(self, per_celltype=True):
        self.per_celltype = per_celltype

    def _feat(self, ids):
        rows = []
        for pid in ids:
            b, c = BAGS[pid], CTS[pid]
            f = [b.mean(0)]
            if self.per_celltype:
                for k in range(len(CT_LIST)):
                    f.append(b[c == k].mean(0) if (c == k).sum() >= 5 else np.full(b.shape[1], np.nan))
            rows.append(np.concatenate(f))
        return np.vstack(rows)

    def fit(self, X, y):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegressionCV
        self.m_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegressionCV(Cs=np.logspace(-4, 2, 20), cv=5, scoring="roc_auc", max_iter=5000))
        self.m_.fit(self._feat(X.index), np.asarray(y))
        return self

    def _score(self, ids):
        return self.m_.predict_proba(self._feat(ids))[:, 1]


# ------------------------------------------------------------------ ABMIL
class _GatedAttn(nn.Module):
    def __init__(self, d, h=64, drop=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        self.V, self.U, self.w = nn.Linear(h, 32), nn.Linear(h, 32), nn.Linear(32, 1)
        self.out = nn.Linear(h, 1)

    def forward(self, x):                                     # x: (B, n, d)
        H = self.enc(x)
        a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))).squeeze(-1)
        a = torch.softmax(a, dim=1)
        return self.out((a.unsqueeze(-1) * H).sum(1)).squeeze(-1)


class ABMIL(_Base):
    def __init__(self, n_cells=512, epochs=150, patience=20, lr=5e-4, wd=1e-3, seed=0, n_test_bags=10):
        self.n_cells, self.epochs, self.patience, self.lr, self.wd = n_cells, epochs, patience, lr, wd
        self.seed, self.n_test_bags = seed, n_test_bags

    def _batch(self, ids, rng):
        return torch.from_numpy(np.stack([_sample(p, self.n_cells, rng) for p in ids])).to(DEV)

    def fit(self, X, y):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        tr, va, ytr, yva = _inner_split(X.index, y, self.seed)
        net = _GatedAttn(BAGS[tr[0]].shape[1]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, bad = np.inf, None, 0
        yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
        for ep in range(self.epochs):
            net.train()
            for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 32)):
                opt.zero_grad(); lossf(net(self._batch(tr[b], rng)), yt[b]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = lossf(net(self._batch(va, rng)), torch.tensor(yva, dtype=torch.float32, device=DEV)).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state); self.net_ = net.eval()
        return self

    @torch.no_grad()
    def _score(self, ids):
        rng = np.random.default_rng(123)
        return np.mean([torch.sigmoid(self.net_(self._batch(ids, rng))).cpu().numpy()
                        for _ in range(self.n_test_bags)], axis=0)


# ------------------------------------------------------------------ MixMIL
def _import_mixmil():
    if "torch_scatter" not in sys.modules:           # only the ragged-bag code path needs torch_scatter;
        stub = types.ModuleType("torch_scatter")     # we use fixed-size bags (tensor path)
        stub.scatter_softmax = stub.segment_add_csr = None
        sys.modules["torch_scatter"] = stub
    sys.path.insert(0, REPOS + "mil_mixmil")
    from mixmil import MixMIL
    return MixMIL


class MixMILModel(_Base):
    def __init__(self, n_cells=512, n_epochs=1000, seed=0, n_test_bags=10):
        self.n_cells, self.n_epochs, self.seed, self.n_test_bags = n_cells, n_epochs, seed, n_test_bags

    def _bags(self, ids, rng):
        return torch.from_numpy(np.stack([_sample(p, self.n_cells, rng) for p in ids]))

    def fit(self, X, y):
        MixMIL = _import_mixmil()
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        ids = list(X.index)
        Xs = self._bags(ids, rng)
        self.mu_, self.sd_ = Xs.reshape(-1, Xs.shape[-1]).mean(0), Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-6
        Xs = (Xs - self.mu_) / self.sd_
        F = torch.ones(len(ids), 1)
        Y = torch.tensor(np.asarray(y), dtype=torch.float32)[:, None]
        self.m_ = MixMIL.init_with_mean_model(Xs, F, Y, likelihood="binomial", n_trials=1)
        self.m_.train(Xs, F, Y, n_epochs=self.n_epochs, batch_size=64, verbose=False)
        self.scaling_ = None
        with torch.no_grad():
            u = self.m_(Xs, n_samples=None, predict=True)
            self.scaling_ = (u.mean(0), u.std(0))
        return self

    @torch.no_grad()
    def _score(self, ids):
        rng = np.random.default_rng(123)
        out = []
        for _ in range(self.n_test_bags):
            Xs = (self._bags(ids, rng) - self.mu_) / self.sd_
            u = self.m_.predict(Xs, scaling=self.scaling_)          # bag effect (N, P)
            logit = u[:, 0] + self.m_.alpha[0, 0]
            out.append(torch.sigmoid(logit).numpy())
        return np.mean(out, axis=0)


# ------------------------------------------------------------------ ScRAT
class ScRATModel(_Base):
    """ScRAT defaults: emb 128, 8 heads, 1 layer, dropout 0.3, lr 3e-3, wd 1e-4, batch 16, 100 epochs, StepLR(50, 0.5),
    500 sampled cells per bag, 20 bags per training donor, cell-type-stratified mixup (alpha 1) of random donor pairs,
    early stopping on validation loss; test = mean over 100 bags (we average probabilities)."""
    def __init__(self, n_cells=500, bags_per_donor=20, augment_num=200, epochs=100, seed=0, n_test_bags=20,
                 lr=3e-3, wd=1e-4):
        self.n_cells, self.bags_per_donor, self.augment_num, self.epochs = n_cells, bags_per_donor, augment_num, epochs
        self.seed, self.n_test_bags, self.lr, self.wd = seed, n_test_bags, lr, wd

    def _mixup_bag(self, p1, p2, lam, rng):
        """Cell-type-stratified mixup as in ScRAT utils.mixups: per cell type, n cells in proportion
        lam*frac1 + (1-lam)*frac2; a cell = lam*cell1 + (1-lam)*cell2 (a missing type uses zeros)."""
        out = []
        for k in np.union1d(np.unique(CTS[p1]), np.unique(CTS[p2])):
            i1, i2 = np.where(CTS[p1] == k)[0], np.where(CTS[p2] == k)[0]
            n = max(int(self.n_cells * (lam * len(i1) / len(CTS[p1]) + (1 - lam) * len(i2) / len(CTS[p2]))), 1)
            a = BAGS[p1][rng.choice(i1, n)] if len(i1) else np.zeros((n, BAGS[p1].shape[1]), np.float32)
            b = BAGS[p2][rng.choice(i2, n)] if len(i2) else np.zeros((n, BAGS[p1].shape[1]), np.float32)
            x = lam * a + (1 - lam) * b
            if not (len(i1) and len(i2)):
                x = x + rng.normal(0, np.sqrt(1e-5), x.shape)
            out.append(x)
        x = np.concatenate(out).astype(np.float32)
        return x[rng.choice(len(x), self.n_cells, replace=len(x) < self.n_cells)]

    def fit(self, X, y):
        sys.path.insert(0, REPOS + "mil_ScRAT")
        from Transformer import TransformerPredictor
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        tr, va, ytr, yva = _inner_split(X.index, y, self.seed)
        # training set: 20 sampled bags per donor + augment_num mixup donors (x 20 bags each, like the originals)
        xs, ys = [], []
        for p, lab in zip(tr, ytr):
            for _ in range(self.bags_per_donor):
                xs.append(_sample(p, self.n_cells, rng)); ys.append(lab)
        for _ in range(self.augment_num):
            i, j = rng.integers(len(tr), size=2); lam = rng.beta(1.0, 1.0)
            for _ in range(self.bags_per_donor // 4):
                xs.append(self._mixup_bag(tr[i], tr[j], lam, rng)); ys.append(lam * ytr[i] + (1 - lam) * ytr[j])
        Xtr = torch.from_numpy(np.stack(xs)); Ytr = torch.tensor(ys, dtype=torch.float32)
        Xva = torch.from_numpy(np.stack([_sample(p, self.n_cells, rng) for p in va for _ in range(5)])).to(DEV)
        Yva = torch.tensor(np.repeat(yva, 5), dtype=torch.float32, device=DEV)
        net = TransformerPredictor(input_dim=Xtr.shape[-1], model_dim=128, num_classes=1, num_heads=8, num_layers=1,
                                   dropout=0.3, input_dropout=0, pca=True, norm_first=False).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        bce = nn.BCEWithLogitsLoss()
        best, best_state = np.inf, None
        for ep in range(self.epochs):
            net.train()
            for b in torch.split(torch.randperm(len(Xtr)), 16):
                opt.zero_grad()
                bce(net(Xtr[b].to(DEV)).squeeze(-1), Ytr[b].to(DEV)).backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = torch.cat([net(c).squeeze(-1) for c in torch.split(Xva, 64)])
                vl = bce(vl, Yva).item()
            if vl < best:
                best, best_state = vl, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state); self.net_ = net.eval()
        return self

    @torch.no_grad()
    def _score(self, ids):
        rng = np.random.default_rng(123)
        out = []
        for _ in range(self.n_test_bags):
            x = torch.from_numpy(np.stack([_sample(p, self.n_cells, rng) for p in ids])).to(DEV)
            out.append(torch.sigmoid(torch.cat([self.net_(c).squeeze(-1) for c in torch.split(x, 64)])).cpu().numpy())
        return np.mean(out, axis=0)


# ------------------------------------------------------------------ CloudPred
class CloudPredModel(_Base):
    """Official cloudpred.train (GMM on training cells -> mixture DensityClassifier -> SGD fine-tuning),
    number of centers chosen on the inner validation donors among `centers`, as the original main.py does."""
    def __init__(self, centers=(5, 10, 20), n_cells=300, seed=0):
        self.centers, self.n_cells, self.seed = centers, n_cells, seed

    def fit(self, X, y):
        if "cloudpred" not in sys.modules:           # skip the package __init__ (CLI parser / argcomplete)
            pkg = types.ModuleType("cloudpred"); pkg.__path__ = [REPOS + "mil_CloudPred/cloudpred"]
            sys.modules["cloudpred"] = pkg
            import importlib
            pkg.utils = importlib.import_module("cloudpred.utils")
            pkg.cloudpred = importlib.import_module("cloudpred.cloudpred")
        import cloudpred
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed); np.random.seed(self.seed)
        tr, va, ytr, yva = _inner_split(X.index, y, self.seed)
        mk = lambda ids, ys: [(_sample(p, self.n_cells, rng), int(l)) for p, l in zip(ids, ys)]
        Xtr, Xva = mk(tr, ytr), mk(va, yva)
        best = None
        for k in self.centers:
            model, res = cloudpred.cloudpred.train(Xtr, Xva, centers=k)
            loss = res["loss"] if isinstance(res, dict) and "loss" in res else -res.get("auc", 0)
            if best is None or loss < best[0]:
                best = (loss, model, k)
        self.m_, self.k_ = best[1], best[2]
        return self

    @torch.no_grad()
    def _score(self, ids):
        rng = np.random.default_rng(123)
        out = []
        for p in ids:
            z = self.m_(torch.Tensor(_sample(p, self.n_cells, rng)))
            out.append(torch.softmax(z, 1)[0, 1].item())
        return np.asarray(out)


# ------------------------------------------------------------------ scMILD
class ScMILDModel(_Base):
    """scMILD (Jeong et al.): official TeacherBranch / StudentBranch / AttentionModule and orthogonal projection loss
    (src/model.py, src/optimizer.py), training loop re-implemented from Optimizer.optimize/optimize_teacher/
    optimize_student with the README defaults (mil_latent 64, lr 1e-3, <= 500 epochs, patience 15 on val loss,
    student every 3 epochs, negative weight 0.3, OPL lambda 0.5 / gamma 0.5 with a 2-component GMM on pseudo-labels).
    The encoder = their EncoderBranch head (3-layer Tanh MLP, fine-tuned) on a frozen latent: here the scVI latent
    instead of their autoencoder's. Bags are subsampled to n_cells per step; the student sees `student_cells` cells."""
    def __init__(self, n_cells=512, epochs=500, patience=15, stu_period=3, neg_weight=0.3, op_lambda=0.5,
                 student_cells=20000, seed=0, n_test_bags=10):
        self.n_cells, self.epochs, self.patience, self.stu_period = n_cells, epochs, patience, stu_period
        self.neg_weight, self.op_lambda, self.student_cells, self.seed = neg_weight, op_lambda, student_cells, seed
        self.n_test_bags = n_test_bags

    def _imports(self):
        import importlib.util
        if "termcolor" not in sys.modules:
            try:
                import termcolor  # noqa: F401
            except ImportError:            # only used for coloured prints in optimizer.py
                tc = types.ModuleType("termcolor"); tc.colored = lambda s, *a, **k: s; sys.modules["termcolor"] = tc
        def load(name):                    # load the file directly: src/__init__ imports modin (data loading only)
            spec = importlib.util.spec_from_file_location(f"scmild_{name}", REPOS + f"mil_scMILD/src/{name}.py")
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
        m, o = load("model"), load("optimizer")
        return m.TeacherBranch, m.StudentBranch, m.AttentionModule, o.OrthogonalProjectionLoss

    def _bag_logits(self, ids, rng):
        out = []
        for p in ids:
            x = torch.from_numpy(_sample(p, self.n_cells, rng)).to(DEV)
            out.append(self.teacher_(self.encoder_(x)))
        return torch.stack(out)

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        Teacher, Student, Attn, OPL = self._imports()
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        tr, va, ytr, yva = _inner_split(X.index, y, self.seed)
        d = BAGS[tr[0]].shape[1]
        self.encoder_ = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d)).to(DEV)
        for m in self.encoder_:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        self.teacher_ = Teacher(input_dims=d, latent_dims=64, attention_module=Attn(L=d, D=d, K=1), num_classes=2,
                                activation_function=nn.Tanh).to(DEV)
        student = Student(input_dims=d, latent_dims=64, num_classes=2, activation_function=nn.Tanh).to(DEV)
        opt_t = torch.optim.Adam(self.teacher_.parameters(), lr=1e-3)
        opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
        opt_e = torch.optim.Adam(self.encoder_.parameters(), lr=1e-3)
        opl = OPL(gamma=0.5, device=DEV)
        bce = lambda prob1, lab: -(lab * torch.log(prob1 + 1e-5) + (1 - lab) * torch.log(1 - prob1 + 1e-5))
        yt = torch.tensor(ytr, dtype=torch.float32, device=DEV); yv = torch.tensor(yva, dtype=torch.float32, device=DEV)
        best, states, bad = np.inf, None, 0
        for ep in range(self.epochs):
            # --- teacher (bag level), updates teacher + encoder
            self.encoder_.train(); self.teacher_.train()
            amin, amax = np.inf, -np.inf
            for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 16)):
                logits = self._bag_logits(tr[b], rng)
                loss = bce(torch.softmax(logits, 1)[:, 1], yt[b]).mean()
                opt_t.zero_grad(); opt_e.zero_grad(); loss.backward(); opt_t.step(); opt_e.step()
            with torch.no_grad():                      # attention-score range, used to normalise pseudo-labels
                for p in tr[:64]:
                    a = self.teacher_.attention_module(self.encoder_(torch.from_numpy(_sample(p, self.n_cells, rng)).to(DEV)))
                    amin, amax = min(amin, a.min().item()), max(amax, a.max().item())
            # --- student (cell level) every stu_period epochs, updates student + encoder
            if ep % self.stu_period == 0:
                student.train()
                cells, lab = [], []
                per = max(1, self.student_cells // len(tr))
                for p, l in zip(tr, ytr):
                    cells.append(_sample(p, per, rng)); lab += [l] * per
                cells = torch.from_numpy(np.concatenate(cells)).to(DEV); lab = torch.tensor(lab, device=DEV)
                for b in torch.split(torch.randperm(len(cells), device=DEV), 256):
                    feat = self.encoder_(cells[b])
                    with torch.no_grad():
                        s = self.teacher_.attention_module(feat).squeeze(0)
                        pseudo = ((s - amin) / (amax - amin + 1e-8)).clamp(1e-5, 1 - 1e-5)
                        pseudo[lab[b] == 0] = 0
                    pred = torch.softmax(student(feat), 1)
                    loss = -torch.mean(self.neg_weight * (1 - pseudo) * torch.log(pred[:, 0] + 1e-5)
                                       + (1 - self.neg_weight) * pseudo * torch.log(pred[:, 1] + 1e-5))
                    pos = lab[b] == 1
                    if pos.sum() > 2 and ep > 0:
                        pl = pseudo[pos].detach().cpu().numpy().reshape(-1, 1)
                        try:
                            gm = GaussianMixture(2, random_state=self.seed).fit(pl)
                            order = np.argsort(np.argsort(gm.means_.ravel()))
                            opl_lab = torch.zeros(len(b), device=DEV)
                            opl_lab[pos] = torch.tensor(order[gm.predict(pl)], dtype=torch.float32, device=DEV)
                        except Exception:
                            opl_lab = (pseudo > 0.5).float() * pos.float()
                        if (opl_lab == 1).sum() > 0:
                            loss = loss + self.op_lambda * opl(feat, opl_lab)
                    opt_s.zero_grad(); opt_e.zero_grad(); loss.backward(); opt_s.step(); opt_e.step()
            # --- validation loss (teacher), early stopping
            self.encoder_.eval(); self.teacher_.eval()
            with torch.no_grad():
                vl = bce(torch.softmax(self._bag_logits(va, rng), 1)[:, 1], yv).mean().item()
            if vl < best:
                best, bad = vl, 0
                states = (copy.deepcopy(self.encoder_.state_dict()), copy.deepcopy(self.teacher_.state_dict()))
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.encoder_.load_state_dict(states[0]); self.teacher_.load_state_dict(states[1])
        self.encoder_.eval(); self.teacher_.eval(); self.n_epochs_ = ep + 1
        return self

    @torch.no_grad()
    def _score(self, ids):
        rng = np.random.default_rng(123)
        return np.mean([torch.softmax(self._bag_logits(ids, rng), 1)[:, 1].cpu().numpy()
                        for _ in range(self.n_test_bags)], axis=0)
