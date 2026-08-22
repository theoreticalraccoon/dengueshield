"""Model 1 - individual dengue screening from clinical/haematological data.

Design decisions:
  * Repeated stratified CV (not a single split): n~1.5k, one split is too noisy.
  * All preprocessing fitted INSIDE the fold (leakage control).
  * Probabilities are calibrated - this is a screening tool, so 0.87 must mean 0.87.
  * Threshold chosen for a target sensitivity, not 0.5. Missing dengue is far more
    costly than a false alarm, and accuracy at 0.5 is the wrong operating point.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# --------------------------------------------------------------------------- MLP
class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), p_drop=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """PyTorch MLP with a sklearn interface, so it drops into the same CV harness."""

    def __init__(self, hidden=(128, 64), p_drop=0.3, lr=1e-3, epochs=200,
                 batch_size=64, weight_decay=1e-4, patience=25, seed=SEED):
        self.hidden, self.p_drop, self.lr = hidden, p_drop, lr
        self.epochs, self.batch_size = epochs, batch_size
        self.weight_decay, self.patience, self.seed = weight_decay, patience, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)
        # internal validation split for early stopping
        idx = np.arange(len(y))
        tr, va = next(StratifiedKFold(5, shuffle=True, random_state=self.seed).split(idx, y))
        Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])
        Xv, yv = torch.tensor(X[va]), torch.tensor(y[va])
        self.model_ = _MLP(X.shape[1], self.hidden, self.p_drop)
        pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
        best, best_state, bad = np.inf, None, 0
        n = len(yt)
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = crit(self.model_(Xt[b]), yt[b])
                loss.backward(); opt.step()
            self.model_.eval()
            with torch.no_grad():
                vl = crit(self.model_(Xv), yv).item()
            sched.step(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(np.asarray(X, dtype=np.float32)))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ------------------------------------------------------------------------ models
def build_models(scale_pos_weight: float = 1.0) -> dict:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    pre = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return {
        "LogisticRegression": Pipeline(pre + [("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "XGBoost": Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("clf", XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.03,
                                                   subsample=0.85, colsample_bytree=0.85,
                                                   min_child_weight=3, reg_lambda=2.0,
                                                   eval_metric="logloss", n_jobs=4,
                                                   random_state=SEED, verbosity=0))]),
        "LightGBM": Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("clf", LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.03,
                                                     subsample=0.85, colsample_bytree=0.85,
                                                     min_child_samples=20, reg_lambda=2.0,
                                                     random_state=SEED, n_jobs=4, verbose=-1))]),
        "PyTorch_MLP": Pipeline(pre + [("clf", TorchMLP())]),
    }


# ----------------------------------------------------------------------- metrics
def metrics_at(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return {"threshold": thr, "accuracy": accuracy_score(y, yhat),
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "balanced_accuracy": (sens + spec) / 2,
            "f1": 2 * ppv * sens / max(ppv + sens, 1e-9),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def threshold_for_sensitivity(y, p, target=0.90) -> float:
    """Highest threshold that still achieves >= target sensitivity.

    Sensitivity is monotonically non-increasing in the threshold, so we walk
    thresholds upward and keep the last one that still meets the target. This
    maximises specificity subject to the sensitivity constraint - the correct
    operating point for a screening tool, where missed cases are the costly error.
    """
    y = np.asarray(y); p = np.asarray(p)
    best = 0.0
    for t in np.unique(p):
        if (p >= t)[y == 1].mean() >= target:
            best = float(t)
        else:
            break
    return best
