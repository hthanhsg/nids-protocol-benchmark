"""Model families.

Four are included, chosen so that the protocol effect can be separated from the
architecture effect. XGBoost is deliberately not a neural model: if it degrades
under protocol changes exactly as the deep models do, the effect is a property of
the evaluation design rather than of deep learning.

Every neural model early-stops on validation macro-F1, never on the test set.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import f1_score

log = logging.getLogger("nidsbench.models")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def val_macro_f1(y_true, y_pred):
    """Macro-F1 restricted to the classes actually present in y_true.

    sklearn's default macro averaging spans the union of labels in y_true and
    y_pred, so a single stray prediction of an absent class adds an F1 of 0 and
    drags the mean down. Under disjoint-entity splits, validation often holds
    only a subset of classes, which made the raw score collapse (0.25 while test
    was 0.82) and caused early stopping to select an essentially untrained
    model. Selection must therefore score only the classes validation can judge.
    """
    present = np.unique(y_true)
    return f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)


def _device():
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _EarlyStopper:
    """Track the best validation macro-F1 and restore those weights at the end.

    `min_epochs` exists because of a failure mode seen under disjoint-entity
    splits: validation and test hold different entities and therefore different
    class mixes, so the validation curve can be flat or even decreasing while
    test performance is fine. Without a floor, the stopper selects epoch 0 and
    the reported score comes from an essentially untrained model. Requiring a
    minimum number of epochs before any checkpoint is kept removes that
    pathology; the `val_uninformative` flag records when it bit.
    """

    def __init__(self, patience, min_delta=1e-4, min_epochs=5):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best = -np.inf
        self.best_state = None
        self.bad = 0
        self.best_epoch = -1

    def step(self, score, model, epoch):
        if epoch < self.min_epochs:
            # Always keep the latest weights until the floor is reached, so the
            # restored model has actually been trained.
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.best, self.best_epoch = score, epoch
            return False
        if score > self.best + self.min_delta:
            self.best, self.bad, self.best_epoch = score, 0, epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _train_torch(model, Xtr, ytr, Xva, yva, n_classes, cfg, name):
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Class weighting keeps rare attack classes from being ignored outright,
    # which would make macro-F1 uninformative for the very classes we care about.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, len(ytr) / (n_classes * np.maximum(counts, 1)), 0.0)
    w = np.clip(w, 0, 50.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    stopper = _EarlyStopper(cfg["patience"], min_epochs=cfg.get("min_epochs", 5))
    history = []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb)

        # Validation must be scored in batches. Running the whole split in one
        # forward pass costs several GB of attention matrices for the
        # transformer and exceeds the CUDA grid limit once validation passes
        # ~65k rows, which surfaces as "invalid configuration argument".
        pv = _predict_torch(model, Xva, batch=cfg["eval_batch"])
        vf1 = val_macro_f1(yva, pv)
        history.append({"epoch": epoch, "train_loss": tot / len(ytr), "val_macro_f1": float(vf1)})
        if stopper.step(vf1, model, epoch):
            log.info("  %s early-stopped at epoch %d (best %d, val macro-F1 %.4f)",
                     name, epoch, stopper.best_epoch, stopper.best)
            break

    stopper.restore(model)
    # If validation never improved past the floor, it gave no usable selection
    # signal. Report it rather than letting it pass silently.
    uninformative = stopper.best_epoch <= stopper.min_epochs and len(history) > stopper.min_epochs + 1
    return model, {"best_epoch": stopper.best_epoch, "epochs_run": len(history),
                   "best_val_macro_f1": float(stopper.best),
                   "val_uninformative": bool(uninformative), "history": history}


def _predict_torch(model, X, batch=8192):
    dev = _device()
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(dev)
            out.append(model(xb).argmax(1).cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out) if out else np.array([], dtype=int)


class MLP(nn.Module if HAS_TORCH else object):
    def __init__(self, d_in, n_classes, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TabTransformer(nn.Module if HAS_TORCH else object):
    """Each feature becomes a token; self-attention learns interactions among them.

    This is the attention branch of the protocol -- a compact stand-in for
    FT-Transformer rather than a reimplementation of it.
    """

    def __init__(self, d_in, n_classes, d_model=32, heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.d_in = d_in
        self.proj = nn.Parameter(torch.randn(d_in, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_in, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.proj.unsqueeze(0) + self.bias.unsqueeze(0)
        t = torch.cat([self.cls.expand(x.size(0), -1, -1), t], dim=1)
        return self.head(self.enc(t)[:, 0])


class AEClassifier(nn.Module if HAS_TORCH else object):
    """Autoencoder pretrained on benign traffic, then fine-tuned as a classifier.

    Represents the unsupervised branch. The reconstruction stage sees benign
    rows only, mirroring how these models are deployed.
    """

    def __init__(self, d_in, n_classes, latent=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, d_in))
        self.head = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, n_classes))
        self.mode = "clf"

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z) if self.mode == "recon" else self.head(z)


def _pretrain_ae(model, Xtr, ytr, cfg):
    dev = _device()
    benign = Xtr[ytr == 0]
    if len(benign) < 100:
        return
    model.mode = "recon"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    crit = torch.nn.MSELoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(benign, dtype=torch.float32)),
        batch_size=cfg["batch_size"], shuffle=True)
    model.train()
    for _ in range(min(15, cfg["max_epochs"])):
        for (xb,) in dl:
            xb = xb.to(dev)
            opt.zero_grad()
            crit(model(xb), xb).backward()
            opt.step()
    model.mode = "clf"


DEFAULT_CFG = {
    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 1024,
    "max_epochs": 100, "patience": 10, "min_epochs": 5, "eval_batch": 8192,
}


def fit_predict(model_name, Xtr, ytr, Xva, yva, Xte, n_classes, seed=42, cfg=None):
    """Train `model_name` and return (test predictions, val predictions, info)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    if model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        # XGBoost requires the training labels to be contiguous 0..k-1. Under
        # temporal and disjoint-entity splits the training fold routinely lacks
        # some classes, leaving gaps that XGBoost rejects outright. Remap to a
        # dense range for fitting and invert afterwards, so the run produces a
        # result instead of failing -- silently losing exactly the hardest cells
        # would bias the model comparison in XGBoost's favour.
        present = np.unique(ytr)
        remap = {c: i for i, c in enumerate(present)}
        inverse = np.array(present)
        ytr_d = np.array([remap[v] for v in ytr], dtype=int)
        yva_d = np.array([remap.get(v, 0) for v in yva], dtype=int)
        k = len(present)
        multi = k > 2
        counts = np.bincount(ytr_d, minlength=k).astype(float)
        w = np.where(counts > 0, len(ytr) / (k * np.maximum(counts, 1)), 1.0)
        sw = w[ytr_d]
        clf = XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="mlogloss" if multi else "logloss",
            early_stopping_rounds=30, random_state=seed, n_jobs=-1,
            objective="multi:softmax" if multi else "binary:logistic",
        )
        clf.fit(Xtr, ytr_d, sample_weight=sw, eval_set=[(Xva, yva_d)], verbose=False)
        info = {"best_epoch": int(getattr(clf, "best_iteration", -1) or -1),
                "epochs_run": int(getattr(clf, "best_iteration", 0) or 0),
                "_model": clf}
        return (inverse[clf.predict(Xte).astype(int)],
                inverse[clf.predict(Xva).astype(int)], info)

    if not HAS_TORCH:
        raise RuntimeError("torch not installed")
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in = Xtr.shape[1]
    if model_name == "mlp":
        model = MLP(d_in, n_classes)
    elif model_name == "transformer":
        model = TabTransformer(d_in, n_classes)
    elif model_name == "autoencoder":
        model = AEClassifier(d_in, n_classes).to(_device())
        _pretrain_ae(model, Xtr, ytr, cfg)
    else:
        raise ValueError(f"unknown model {model_name}")

    model, info = _train_torch(model, Xtr, ytr, Xva, yva, n_classes, cfg, model_name)
    eb = cfg["eval_batch"]
    info["_model"] = model
    return _predict_torch(model, Xte, eb), _predict_torch(model, Xva, eb), info


# egraphsage is dispatched separately in run.py because it needs graph structure
# rather than a feature matrix.
TABULAR_MODELS = ["xgboost", "mlp", "transformer", "autoencoder"]
ALL_MODELS = TABULAR_MODELS + ["egraphsage"]