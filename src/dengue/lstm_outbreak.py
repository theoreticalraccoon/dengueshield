"""Spatiotemporal LSTM for dengue outbreak forecasting (PyTorch).

Each sample is one municipality's last SEQ_LEN weeks of dynamic covariates
(cases, climate, vegetation, ENSO) plus its static context (land cover,
sanitation, demography). The LSTM encodes the temporal window; static features
are concatenated to the final hidden state before the classifier head.

Sequences never cross a municipality boundary or a gap in the weekly series,
and the label is taken HORIZON weeks after the end of the window.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEQ_LEN = 12


def make_sequences(df: pd.DataFrame, dyn: list[str], stat: list[str],
                   seq_len: int = SEQ_LEN):
    """Build (N, seq_len, n_dyn), (N, n_stat), (N,) arrays plus the row index of each label."""
    Xs, Ss, ys, idx = [], [], [], []
    for _, g in df.groupby("codigo_ibge", sort=False):
        g = g.sort_values("data_iniSE")
        # contiguous weekly blocks only
        gap = g["data_iniSE"].diff().dt.days.fillna(7).ne(7).cumsum()
        for _, blk in g.groupby(gap):
            if len(blk) < seq_len:
                continue
            D = blk[dyn].to_numpy(dtype=np.float32)
            S = blk[stat].to_numpy(dtype=np.float32)
            Y = blk["y"].to_numpy(dtype=np.float32)
            rid = blk.index.to_numpy()
            for i in range(seq_len - 1, len(blk)):
                Xs.append(D[i - seq_len + 1: i + 1])
                Ss.append(S[i]); ys.append(Y[i]); idx.append(rid[i])
    if not Xs:
        return (np.empty((0, seq_len, len(dyn)), np.float32),
                np.empty((0, len(stat)), np.float32), np.empty(0, np.float32), np.empty(0, int))
    return np.stack(Xs), np.stack(Ss), np.array(ys, np.float32), np.array(idx)


class OutbreakLSTM(nn.Module):
    def __init__(self, n_dyn: int, n_stat: int, hidden: int = 128,
                 layers: int = 2, p_drop: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_dyn, hidden, num_layers=layers, batch_first=True,
                            dropout=p_drop if layers > 1 else 0.0, bidirectional=False)
        self.static = nn.Sequential(nn.Linear(n_stat, 64), nn.ReLU(), nn.Dropout(p_drop))
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1))

    def forward(self, x_seq, x_stat):
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]                       # last timestep encoding
        s = self.static(x_stat)
        return self.head(torch.cat([h, s], dim=1)).squeeze(-1)


def train_lstm(Xtr, Str, ytr, Xva, Sva, yva, epochs=40, bs=512, lr=1e-3,
               patience=6, device=None, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = OutbreakLSTM(Xtr.shape[2], Str.shape[1]).to(device)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                         dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Str), torch.tensor(ytr)),
                    batch_size=bs, shuffle=True, drop_last=True)
    Xv = torch.tensor(Xva, device=device)
    Sv = torch.tensor(Sva, device=device)
    from sklearn.metrics import average_precision_score
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb, sb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv, Sv)).cpu().numpy()
        ap = average_precision_score(yva, pv) if len(np.unique(yva)) > 1 else 0.0
        sched.step(-ap)
        if ap > best + 1e-5:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"    epoch {ep:02d}  val PR-AUC={ap:.4f}{'  *' if bad==0 else ''}", flush=True)
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, X, S, device=None, bs=4096):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], device=device); sb = torch.tensor(S[i:i+bs], device=device)
        out.append(torch.sigmoid(model(xb, sb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
