# 2. The Brazil horizon is 2 weeks, defined in one place

Date: 2026-08-23
Status: Accepted (code); the retrain it implies is outstanding

## Context

`src/dengue/model2_outbreak.py` declared `HORIZON = 4`. Seven scripts overrode it
locally to `2` — `shap_model2.py`, `spatial_and_ablation.py`,
`calibration_and_errors.py`, `leakage_audit.py`, `verify_fixes.py`,
`transfer_srilanka.py`, `finalize_srilanka.py` — three of them commenting that 2 is
"the flagship" configuration.

The two scripts that *save models* declared nothing and silently inherited 4:
`train_model2.py` and `train_lstm.py`. Loading the bundle confirms it:

    models/model2_outbreak.joblib   horizon = 4
    models/srilanka_outbreak.joblib horizon = 2

So the repo holds two families of Brazil artifact at two different horizons:

| Artifact | Producer | Horizon | LightGBM PR-AUC |
|---|---|---|---|
| `model2_summary.json`, `model2_results.csv`, `model2_feature_importance.csv`, `models/model2_outbreak.joblib`, `models/model2_lstm.pt` | `train_model2.py`, `train_lstm.py` | **4** | 0.9166 |
| `model2_calibration_errors.json`, `model2_spatial_ablation.json`, `model2_robustness.json`, `leakage_audit.json`, `lag_fix_impact.json` | the seven overriding scripts | **2** | 0.9614 |

The headline result quoted in `CLAUDE.md`, `README.md` and the app's About screen —
task C, PR-AUC **0.961** — comes from the horizon-2 family. The saved model does not.

Nothing user-facing is wrong today: `app.py` never loads `model2_outbreak.joblib`.
The defect is that the released model and the published evidence describe different
experiments, and nothing made that visible.

## Decision

`BR_HORIZON = 2` lives in `src/dengue/config.py`. `model2_outbreak` re-exports it as
`HORIZON` so existing callers are unaffected, and all seven local overrides now
derive from it. The value can no longer diverge, because there is only one of it.

The same applies to `BR_INC`, `SL_INC`, `SL_HORIZON`, `SEED` and the split years.

## Consequences

- Every script now agrees on horizon 2, including `train_model2.py` and
  `train_lstm.py`, which previously trained at 4.
- **The committed horizon-4 artifacts are now stale with respect to the code that
  produced them.** Re-running `train_model2.py` will retrain at horizon 2 and
  overwrite `model2_summary.json`, `model2_results.csv` and
  `model2_feature_importance.csv` — currently horizon-4 numbers — as well as the
  bundle. That is the intended end state, but it rewrites published evidence and
  is therefore a deliberate, separately reviewed act, not a side effect of this
  change.
- `train_lstm.py` is in the same position and additionally needs torch and the
  569 MB Brazil parquet, so the LSTM artifacts (`model2_lstm.pt`,
  `model2_lstm.json`, `model2_lstm_test_preds.npz`) remain horizon-4 until it is
  re-run. `model2_headtohead.csv` compares the two and is a snapshot of that older
  state.
- Until the retrain happens, this ADR is the record of which artifacts sit at which
  horizon. Do not quote `model2_summary.json` and the horizon-2 reports in the same
  table without saying so.

## Alternatives rejected

**Set `BR_HORIZON = 4` to match the saved bundle.** It would make the code agree
with the models rather than the evidence, and would silently redefine the flagship
14-day claim as a 28-day one. The horizon-2 reports are the ones the project's
conclusions rest on.

**Leave the override in each script.** That is the state that produced the
divergence.
