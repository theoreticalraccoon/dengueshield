"""Brazil -> Sri Lanka transfer experiment for the outbreak model.

Four conditions, all evaluated on the SAME held-out Sri Lankan test period:

  1. Persistence           - assume the next fortnight looks like this week
  2. Brazil -> Sri Lanka   - zero-shot, no Sri Lankan data seen at all
  3. Sri Lanka only        - trained from scratch on Sri Lankan data
  4. Brazil + fine-tuned   - Brazil model, boosting continued on Sri Lankan data

Only features that genuinely exist in both countries are used. Brazil's MODIS,
land-cover and sanitation layers have no Sri Lankan equivalent here, so they are
dropped rather than imputed - a transferred model must not depend on inputs the
target country cannot supply.
"""
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, precision_score, f1_score, brier_score_loss,
                             confusion_matrix)

from dengue.model2_outbreak import load_panel, build_supervised
from dengue.srilanka import build_panel, add_features, COMMON_FEATURES

HORIZON = 2                 # 14-day early warning - the flagship configuration
BR_INC = 100.0              # Brazil epidemic threshold, per 100k per week
TRAIN_END, VAL_END = 2021, 2023

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63,
              min_child_samples=40, subsample=0.85, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, seed=42)


def metrics(y, p, thr, name):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"condition": name, "threshold": float(thr),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, yh),
            "recall": recall_score(y, yh, zero_division=0),
            "precision": precision_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "specificity": tn / max(tn + fp, 1),
            "brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def tune_thr(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 200)))
    return max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in qs)[1]


def split(df, col="anio"):
    return (df[df[col] <= TRAIN_END], df[(df[col] > TRAIN_END) & (df[col] <= VAL_END)],
            df[df[col] > VAL_END])


# ----------------------------------------------------------------- Sri Lanka data
print("building Sri Lanka district-week panel ...", flush=True)
sl_panel = build_panel()
print(f"  panel: {sl_panel.shape}  districts={sl_panel.district.nunique()}  "
      f"{sl_panel.week_start.min().date()} -> {sl_panel.week_start.max().date()}", flush=True)

inc = sl_panel.p_inc100k
print(f"  weekly incidence per 100k: median={inc.median():.1f} p90={inc.quantile(.9):.1f} "
      f"p99={inc.quantile(.99):.1f} max={inc.max():.1f}")
frac_over_br = float((inc >= BR_INC).mean())
print(f"  fraction of district-weeks >= {BR_INC:.0f}/100k (Brazil threshold): {frac_over_br:.4f}")

# Brazil's absolute threshold may be unreachable in Sri Lanka. If so, also run a
# Sri Lanka-calibrated threshold chosen to match Brazil's outbreak PREVALENCE, so
# the two countries pose a comparably hard problem.
sl_thresholds = {"brazil_absolute_100": BR_INC}
if frac_over_br < 0.04:
    q = float(np.quantile(inc[inc > 0], 0.90))
    sl_thresholds["srilanka_calibrated"] = round(q, 1)
    print(f"  -> Brazil threshold is near-degenerate here; adding a Sri Lanka-calibrated "
          f"threshold of {q:.1f}/100k")

# ------------------------------------------------------------------- Brazil data
print("\nbuilding Brazil panel (common features only) ...", flush=True)
br = build_supervised(load_panel(), horizon=HORIZON, outbreak_inc=BR_INC)
feats = [c for c in COMMON_FEATURES if c in br.columns]
missing = [c for c in COMMON_FEATURES if c not in br.columns]
print(f"  common features used: {len(feats)}" + (f"  (missing in Brazil: {missing})" if missing else ""))
br_tr, br_va, br_te = split(br)
print(f"  Brazil train={len(br_tr)} val={len(br_va)}  outbreak_rate(train)={br_tr.y.mean():.4f}")

# Brazil model on the shared feature space
spw = (br_tr.y.values == 0).sum() / max((br_tr.y.values == 1).sum(), 1)
br_ds = lgb.Dataset(br_tr[feats], label=br_tr.y.values, free_raw_data=False)
br_model = lgb.train({**PARAMS, "scale_pos_weight": spw}, br_ds, num_boost_round=700)
br_val_p = br_model.predict(br_va[feats])
print(f"  Brazil val PR-AUC (sanity) = {average_precision_score(br_va.y.values, br_val_p):.4f}")

# ------------------------------------------------------------------ experiments
all_rows = []
for tname, thr_inc in sl_thresholds.items():
    print(f"\n{'='*78}\nSRI LANKA outbreak threshold: {tname}  (>= {thr_inc}/100k)")
    sl = add_features(sl_panel, horizon=HORIZON, outbreak_inc=thr_inc)
    sl_tr, sl_va, sl_te = split(sl)
    print(f"  train={len(sl_tr)} val={len(sl_va)} test={len(sl_te)}  "
          f"outbreak_rate: train={sl_tr.y.mean():.4f} test={sl_te.y.mean():.4f}")
    if len(sl_te) == 0 or sl_te.y.sum() < 10:
        print("  test split degenerate - skipping")
        continue
    yte = sl_te.y.values
    trivial = max(yte.mean(), 1 - yte.mean())
    print(f"  trivial 'never an outbreak' accuracy = {trivial:.4f}")
    rows = []

    # 1. persistence
    rows.append(metrics(yte, sl_te.baseline_persistence.values, 0.5, "1_persistence"))

    # 2. zero-shot Brazil -> Sri Lanka
    p_va = br_model.predict(sl_va[feats])
    p_te = br_model.predict(sl_te[feats])
    rows.append(metrics(yte, p_te, tune_thr(sl_va.y.values, p_va), "2_brazil_zeroshot"))

    # 3. Sri Lanka only
    spw_sl = (sl_tr.y.values == 0).sum() / max((sl_tr.y.values == 1).sum(), 1)
    sl_ds = lgb.Dataset(sl_tr[feats], label=sl_tr.y.values, free_raw_data=False)
    sl_model = lgb.train({**PARAMS, "scale_pos_weight": spw_sl}, sl_ds, num_boost_round=500)
    p_va = sl_model.predict(sl_va[feats]); p_te = sl_model.predict(sl_te[feats])
    rows.append(metrics(yte, p_te, tune_thr(sl_va.y.values, p_va), "3_srilanka_only"))

    # 4. Brazil pretrained + Sri Lanka fine-tuning (continue boosting on SL data)
    ft = lgb.train({**PARAMS, "scale_pos_weight": spw_sl, "learning_rate": 0.02},
                   sl_ds, num_boost_round=300, init_model=br_model)
    p_va = ft.predict(sl_va[feats]); p_te = ft.predict(sl_te[feats])
    rows.append(metrics(yte, p_te, tune_thr(sl_va.y.values, p_va), "4_brazil_finetuned"))

    for r in rows:
        r["threshold_name"] = tname
        r["outbreak_inc"] = thr_inc
        r["test_prevalence"] = float(yte.mean())
        r["trivial_accuracy"] = float(trivial)
    all_rows += rows
    df = pd.DataFrame(rows)
    print(df[["condition", "roc_auc", "pr_auc", "accuracy", "recall",
              "precision", "f1"]].round(4).to_string(index=False))

res = pd.DataFrame(all_rows)
res.to_csv("reports/srilanka_transfer.csv", index=False)
json.dump(all_rows, open("reports/srilanka_transfer.json", "w"), indent=2, default=float)
sl_panel.to_parquet("data/processed/srilanka_panel.parquet", index=False)
print("\nsaved reports/srilanka_transfer.csv and data/processed/srilanka_panel.parquet")
