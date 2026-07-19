# Benchmark methodology

## Scope

This is a research benchmark, not evidence of diagnostic performance. The labelled `structured_endometriosis_data.csv` has unconfirmed provenance and is the only training source.

## Procedure

1. Create standard BMI categories in the labelled data and NHANES reference data.
2. Filter NHANES to women aged 18–49 with valid `RHQ031` and measured BMI.
3. Use `WTINT2YR` survey weights to form the NHANES BMI-category × irregularity distribution.
4. Form calibration ratios relative to the labelled data, cap raw weights at 5× their positive median, normalize to mean one, and train a weighted model.
5. Train both models with a fixed stratified 80/20 split (`random_state=42`) and `HistGradientBoostingClassifier(max_depth=4, max_iter=150)`.

## Actual run results

| Metric | Raw model | Calibration-weighted model |
|---|---:|---:|
| Held-out ROC AUC | 0.6354 | 0.6309 |

| Correction effect | Value |
|---|---:|
| Mean absolute prediction divergence | 0.0660 |
| Maximum divergence | 0.3306 |
| Test patients shifting by > 0.15 | 8.8% |
| NHANES complete reference respondents | 1,206 |
| Rows whose raw calibration weight was capped | 29.0% |

The corrected model is not expected to improve AUC on a held-out sample drawn from the same labelled population. Its value here is to surface when predictions are sensitive to a documented mismatch with the reference population. The saved `artifacts/training_summary.json` is the source of truth for future runs.
