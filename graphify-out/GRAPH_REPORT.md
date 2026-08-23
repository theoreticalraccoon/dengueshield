# Graph Report - DengueShield  (2026-08-22)

## Corpus Check
- 112 files · ~70,940 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 949 nodes · 1143 edges · 81 communities (69 shown, 12 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 72 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Outbreak feature vocabulary
- Calibration and LSTM code
- Frozen v2 manifest
- Project concepts and findings
- v1 baseline manifest
- Clinical feature vocabulary
- Dataset loaders (frozen)
- SHAP driver outputs
- Ablation module (frozen)
- Screening metric vocabulary
- Calibration error outputs
- LightGBM hyperparameters
- WER PDF parser
- Sri Lanka panel (frozen)
- Brazil outbreak pipeline
- Data refresh pipeline
- Screening model module
- Streamlit application
- Operating point metrics
- Complication model metrics
- Sri Lanka panel builder
- Environment manifest
- Robustness sweep results
- Ablation module
- Dataset integrity audit
- Model bundle schema
- Dataset loaders
- Sensitivity operating points
- Screening model bundle
- LSTM outbreak model
- Tuned classifier params
- Model provenance records
- Release freezing
- Release freezing (frozen)
- Spatial and ablation results
- Emergence experiment
- Streamlit app (frozen)
- Calibrator parameters
- Metrics index
- calibration_and_errors.py
- srilanka_outbreak.joblib
- keys
- leakage_audit.py
- robustness_model2.py
- spatial_and_ablation.py
- transfer_srilanka.py
- finalize_emergence_srilanka.py
- files
- train_model2.py
- dataset_audit.csv
- model1_cv_results.csv
- model1_final.json
- model1_oof_calibrated.npy
- model1_oof.npz
- model1_operating_points.json
- model1_summary.json
- model2_feature_importance.csv
- model2_headtohead.csv
- model2_headtohead.json
- model2_lstm.json
- model2_lstm.pt
- model2_lstm_scalers.joblib
- model2_lstm_test_preds.npz
- model2_outbreak.joblib
- model2_results.csv
- model2_robustness.json
- model2_summary.json
- transfer_auc.csv
- peds_complications.joblib
- repeated_oof()
- transfer_test.py
- verify_fixes.py
- Trivial baseline
- finalize_model1.py
- finalize_peds.py
- finalize_srilanka.py
- shap_model2.py

## God Nodes (most connected - your core abstractions)
1. `features` - 59 edges
2. `features` - 36 edges
3. `top_features` - 31 edges
4. `features` - 23 edges
5. `hyperparameters` - 22 edges
6. `files` - 21 edges
7. `features` - 21 edges
8. `metrics` - 15 edges
9. `peds_final` - 15 edges
10. `Outbreak continuation` - 14 edges

## Surprising Connections (you probably didn't know these)
- `SHAP driver chart` --depicts--> `SHAP attribution`  [INFERRED]
  reports/model2_shap_drivers.png → REPORT.md
- `SHAP summary plot` --depicts--> `SHAP attribution`  [INFERRED]
  reports/model2_shap_summary.png → REPORT.md
- `Reliability diagram` --depicts--> `Isotonic calibration`  [INFERRED]
  reports/model2_reliability.png → REPORT.md
- `Tuned for sensitivity` --governs--> `Dengue screening`  [INFERRED]
  CLAUDE.md → REPORT.md
- `Tuned for sensitivity` --governs--> `Complication risk`  [INFERRED]
  CLAUDE.md → REPORT.md

## Import Cycles
- None detected.

## Communities (81 total, 12 thin omitted)

### Community 0 - "Outbreak feature vocabulary"
Cohesion: 0.05
Nodes (60): features, features, area_km2, casos, casos_growth_1, casos_growth_4, casos_lag_1, casos_lag_2 (+52 more)

### Community 1 - "Calibration and LSTM code"
Cohesion: 0.06
Nodes (32): ece(), Model 2: calibration assessment + structured error analysis. Calibration…, Expected Calibration Error over equal-count bins., make_sequences(), OutbreakLSTM, predict(), DataFrame, no_grad (+24 more)

### Community 2 - "Frozen v2 manifest"
Cohesion: 0.04
Nodes (47): accuracy, brier, note, population, pr_auc, question, roc_auc, sensitivity (+39 more)

### Community 3 - "Project concepts and findings"
Cohesion: 0.06
Nodes (45): Persistence baseline, Streamlit application, Scheduled refresh workflow, experiments/emergence_v1, frozen/v2_final, Weekly data refresh, WER PDF parser, Four separate tasks (+37 more)

### Community 4 - "v1 baseline manifest"
Cohesion: 0.05
Nodes (42): chance_pr_auc, shuffled_label_pr_auc, shuffled_label_roc_auc, frozen_at, accuracy, pr_auc, recall, trivial_accuracy (+34 more)

### Community 5 - "Clinical feature vocabulary"
Cohesion: 0.05
Nodes (42): features, features, ABDOMINAL PAIN, Age, ALT, BLEEDING, BP AT ADMISSION, BREATHLESSNESS (+34 more)

### Community 6 - "Dataset loaders (frozen)"
Cohesion: 0.07
Nodes (26): _clin_ratios(), load_bd_comprehensive(), load_bd_structured(), load_hematology_1523(), load_vitals_1003(), DataFrame, Loaders for every candidate dengue dataset, returning (X, y, meta)., Clinically motivated haematological ratios used in dengue literature. (+18 more)

### Community 7 - "SHAP driver outputs"
Cohesion: 0.06
Nodes (34): shap, drivers, horizon_weeks, top_features, area_km2, casos, casos_growth_1, casos_growth_4 (+26 more)

### Community 8 - "Ablation module (frozen)"
Cohesion: 0.12
Nodes (27): full_metrics(), load_peds(), _model(), nested_oof(), DataFrame, ndarray, Feature-group ablation: how much does each category of information add? Answers…, Highest threshold still achieving >= target sensitivity (screening operating… (+19 more)

### Community 9 - "Screening metric vocabulary"
Cohesion: 0.09
Nodes (29): accuracy, balanced_accuracy, brier, f1, npv, ppv, pr_auc, roc_auc (+21 more)

### Community 10 - "Calibration error outputs"
Cohesion: 0.07
Nodes (28): brier, ece, pr_auc, roc_auc, FN, FP, TN, TP (+20 more)

### Community 11 - "LightGBM hyperparameters"
Cohesion: 0.12
Nodes (24): boosting_type, class_weight, colsample_bytree, importance_type, learning_rate, max_depth, min_child_samples, min_child_weight (+16 more)

### Community 12 - "WER PDF parser"
Cohesion: 0.11
Nodes (22): date, district_rows(), fetch(), list_reports(), _month(), _name_of(), parse(), DataFrame (+14 more)

### Community 13 - "Sri Lanka panel (frozen)"
Cohesion: 0.14
Nodes (16): add_features(), build_panel(), _fetch_one(), fetch_weather(), _from_nasa_power(), load_cases(), DataFrame, _rain_days() (+8 more)

### Community 14 - "Brazil outbreak pipeline"
Cohesion: 0.15
Nodes (18): build_supervised(), _date_exact_lag(), feature_columns(), load_panel(), DataFrame, ndarray, Path, Series (+10 more)

### Community 15 - "Data refresh pipeline"
Cohesion: 0.21
Nodes (16): extend_weather(), fetch_hub(), harvest_wer(), log(), main(), merge(), DataFrame, Refresh Sri Lankan surveillance data and regenerate the outbreak forecasts. Two… (+8 more)

### Community 16 - "Screening model module"
Cohesion: 0.15
Nodes (9): build_models(), _MLP, BaseEstimator, ClassifierMixin, Model 1 - individual dengue screening from clinical/haematological data. Design…, Highest threshold that still achieves >= target sensitivity. Sensitivity is…, PyTorch MLP with a sklearn interface, so it drops into the same CV harness., threshold_for_sensitivity() (+1 more)

### Community 17 - "Streamlit application"
Cohesion: 0.14
Nodes (10): build_cbc_row(), load_models(), load_reports(), probability_bar(), cache_data, cache_resource, DengueShield - dengue screening + outbreak early warning. 1. Patient assessment…, Fill any unsupplied CBC field from the cohort median, then derive ratios. (+2 more)

### Community 18 - "Operating point metrics"
Cohesion: 0.13
Nodes (15): accuracy, balanced_accuracy, brier, fn, fp, npv, ppv, pr_auc (+7 more)

### Community 19 - "Complication model metrics"
Cohesion: 0.13
Nodes (15): peds_final, accuracy, balanced_accuracy, brier, fn, fp, npv, ppv (+7 more)

### Community 20 - "Sri Lanka panel builder"
Cohesion: 0.23
Nodes (14): add_features(), build_panel(), _fetch_one(), fetch_weather(), _from_nasa_power(), load_cases(), DataFrame, _rain_days() (+6 more)

### Community 21 - "Environment manifest"
Cohesion: 0.14
Nodes (14): environment, packages, platform, python, duckdb==1.5.5, lightgbm==4.7.0, numpy==2.4.6, pandas==3.0.5 (+6 more)

### Community 22 - "Robustness sweep results"
Cohesion: 0.14
Nodes (14): model2_robustness, horizons, rolling_origin, shuffled_control, thresholds, accuracy, f1, n_test (+6 more)

### Community 23 - "Ablation module"
Cohesion: 0.21
Nodes (13): full_metrics(), load_peds(), _model(), nested_oof(), DataFrame, ndarray, Feature-group ablation: how much does each category of information add? Answers…, Highest threshold still achieving >= target sensitivity (screening operating… (+5 more)

### Community 24 - "Dataset integrity audit"
Cohesion: 0.27
Nodes (13): audit(), _numeric(), permutation_null(), DataFrame, ndarray, Series, range_overlap(), Data integrity auditing: detect synthetic / label-leaking datasets. Public… (+5 more)

### Community 25 - "Model bundle schema"
Cohesion: 0.24
Nodes (13): keys, keys, keys, keys, features, groups, horizon, metrics (+5 more)

### Community 26 - "Dataset loaders"
Cohesion: 0.17
Nodes (12): _clin_ratios(), load_bd_comprehensive(), load_bd_structured(), load_hematology_1523(), load_vitals_1003(), DataFrame, Loaders for every candidate dengue dataset, returning (X, y, meta)., Clinically motivated haematological ratios used in dengue literature. (+4 more)

### Community 27 - "Sensitivity operating points"
Cohesion: 0.17
Nodes (12): accuracy, balanced_accuracy, brier, f1, npv, ppv, pr_auc, roc_auc (+4 more)

### Community 28 - "Screening model bundle"
Cohesion: 0.18
Nodes (11): accuracy@0.5, accuracy@sens90, brier, pr_auc, roc_auc, sensitivity@sens90, specificity@sens90, metrics_nested_cv (+3 more)

### Community 29 - "LSTM outbreak model"
Cohesion: 0.20
Nodes (8): make_sequences(), OutbreakLSTM, predict(), DataFrame, no_grad, Spatiotemporal LSTM for dengue outbreak forecasting (PyTorch). Each sample is…, Build (N, seq_len, n_dyn), (N, n_stat), (N,) arrays plus the row index of each…, train_lstm()

### Community 30 - "Tuned classifier params"
Cohesion: 0.22
Nodes (9): clf__colsample_bytree, clf__learning_rate, clf__min_child_samples, clf__n_estimators, clf__num_leaves, clf__reg_lambda, clf__subsample, clf__subsample_freq (+1 more)

### Community 31 - "Model provenance records"
Cohesion: 0.22
Nodes (9): error, horizon, n_features, outbreak_inc, threshold, model_provenance, model2_lstm.pt, model2_lstm_scalers.joblib (+1 more)

### Community 32 - "Release freezing"
Cohesion: 0.29
Nodes (5): describe(), Path, Create frozen/v2_final/ - the immutable release for the final report. Captures…, Pull feature list, threshold and hyperparameters out of a saved bundle., sha256()

### Community 33 - "Release freezing (frozen)"
Cohesion: 0.29
Nodes (5): describe(), Path, Create frozen/v2_final/ - the immutable release for the final report. Captures…, Pull feature list, threshold and hyperparameters out of a saved bundle., sha256()

### Community 34 - "Spatial and ablation results"
Cohesion: 0.25
Nodes (8): environment_alone_vs_persistence, environment_gain_over_historical, historical_gain_over_persistence, model2_spatial_ablation, ablation_summary, information_ablation, spatial, state_folds

### Community 35 - "Emergence experiment"
Cohesion: 0.38
Nodes (6): build_emergence(), fit_eval(), EXPERIMENT: emerging-outbreak detection (exploratory extension to frozen v2).…, Label the TRANSITION into outbreak, among places not currently in outbreak., score(), tune_thr()

### Community 36 - "Streamlit app (frozen)"
Cohesion: 0.29
Nodes (5): load_models(), load_reports(), cache_data, cache_resource, DengueShield - dengue screening, complication risk, and outbreak early warning.…

### Community 37 - "Calibrator parameters"
Cohesion: 0.48
Nodes (7): cv, ensemble, estimator, method, n_jobs, hyperparameters, hyperparameters

### Community 38 - "Metrics index"
Cohesion: 0.29
Nodes (7): metrics, ablation_peds, dataset_audit, model1_final, model2_headtohead, srilanka_transfer, trivial_baseline

### Community 39 - "calibration_and_errors.py"
Cohesion: 0.33
Nodes (3): ece(), Model 2: calibration assessment + structured error analysis. Calibration…, Expected Calibration Error over equal-count bins.

### Community 40 - "srilanka_outbreak.joblib"
Cohesion: 0.33
Nodes (6): srilanka_outbreak.joblib, horizon, n_features, n_trees, outbreak_inc, threshold

### Community 41 - "keys"
Cohesion: 0.40
Nodes (5): keys, imp_d, imp_s, sc_d, sc_s

### Community 42 - "leakage_audit.py"
Cohesion: 0.40
Nodes (3): evaluate(), Explicit temporal-leakage audit for the outbreak pipeline. Documents, for a…, Re-fit with all features lagged by `delay_weeks`, target unchanged.

### Community 43 - "robustness_model2.py"
Cohesion: 0.60
Nodes (4): evaluate(), mk(), Verification suite for Model 2. A single train/test split can flatter a model.…, tune_thr()

### Community 47 - "files"
Cohesion: 0.50
Nodes (4): files, model1_screening.joblib, bytes, sha256

### Community 49 - "dataset_audit.csv"
Cohesion: 0.67
Nodes (3): bytes, sha256, dataset_audit.csv

### Community 50 - "model1_cv_results.csv"
Cohesion: 0.67
Nodes (3): model1_cv_results.csv, bytes, sha256

### Community 51 - "model1_final.json"
Cohesion: 0.67
Nodes (3): model1_final.json, bytes, sha256

### Community 52 - "model1_oof_calibrated.npy"
Cohesion: 0.67
Nodes (3): model1_oof_calibrated.npy, bytes, sha256

### Community 53 - "model1_oof.npz"
Cohesion: 0.67
Nodes (3): model1_oof.npz, bytes, sha256

### Community 54 - "model1_operating_points.json"
Cohesion: 0.67
Nodes (3): model1_operating_points.json, bytes, sha256

### Community 55 - "model1_summary.json"
Cohesion: 0.67
Nodes (3): model1_summary.json, bytes, sha256

### Community 56 - "model2_feature_importance.csv"
Cohesion: 0.67
Nodes (3): model2_feature_importance.csv, bytes, sha256

### Community 57 - "model2_headtohead.csv"
Cohesion: 0.67
Nodes (3): model2_headtohead.csv, bytes, sha256

### Community 58 - "model2_headtohead.json"
Cohesion: 0.67
Nodes (3): model2_headtohead.json, bytes, sha256

### Community 59 - "model2_lstm.json"
Cohesion: 0.67
Nodes (3): model2_lstm.json, bytes, sha256

### Community 60 - "model2_lstm.pt"
Cohesion: 0.67
Nodes (3): model2_lstm.pt, bytes, sha256

### Community 61 - "model2_lstm_scalers.joblib"
Cohesion: 0.67
Nodes (3): model2_lstm_scalers.joblib, bytes, sha256

### Community 62 - "model2_lstm_test_preds.npz"
Cohesion: 0.67
Nodes (3): model2_lstm_test_preds.npz, bytes, sha256

### Community 63 - "model2_outbreak.joblib"
Cohesion: 0.67
Nodes (3): model2_outbreak.joblib, bytes, sha256

### Community 64 - "model2_results.csv"
Cohesion: 0.67
Nodes (3): model2_results.csv, bytes, sha256

### Community 65 - "model2_robustness.json"
Cohesion: 0.67
Nodes (3): model2_robustness.json, bytes, sha256

### Community 66 - "model2_summary.json"
Cohesion: 0.67
Nodes (3): model2_summary.json, bytes, sha256

### Community 67 - "transfer_auc.csv"
Cohesion: 0.67
Nodes (3): transfer_auc.csv, bytes, sha256

### Community 68 - "peds_complications.joblib"
Cohesion: 0.67
Nodes (3): peds_complications.joblib, n_features, threshold_sens90

## Knowledge Gaps
- **386 isolated node(s):** `frozen_at`, `status`, `dataset`, `definition`, `n_rows_gbm` (+381 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `model_provenance` connect `Model provenance records` to `srilanka_outbreak.joblib`, `Frozen v2 manifest`, `Screening model bundle`, `peds_complications.joblib`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `metrics` connect `Metrics index` to `Frozen v2 manifest`, `Spatial and ablation results`, `SHAP driver outputs`, `Screening metric vocabulary`, `Calibration error outputs`, `Complication model metrics`, `Robustness sweep results`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `model2_outbreak.joblib` connect `Model provenance records` to `Outbreak feature vocabulary`, `Model bundle schema`, `LightGBM hyperparameters`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **What connects `frozen_at`, `status`, `dataset` to the rest of the system?**
  _386 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Outbreak feature vocabulary` be split into smaller, more focused modules?**
  _Cohesion score 0.05254237288135593 - nodes in this community are weakly interconnected._
- **Should `Calibration and LSTM code` be split into smaller, more focused modules?**
  _Cohesion score 0.061224489795918366 - nodes in this community are weakly interconnected._
- **Should `Frozen v2 manifest` be split into smaller, more focused modules?**
  _Cohesion score 0.041666666666666664 - nodes in this community are weakly interconnected._