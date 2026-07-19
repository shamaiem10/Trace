# Benchmark methodology

## Scope

This is a research benchmark, not evidence of diagnostic performance. The labelled `structured_endometriosis_data.csv` has unconfirmed provenance and is the only training-label source.

## Reproducible procedure

1. Create standard BMI categories in the labelled data and NHANES reference data.
2. Filter NHANES to women aged 18-49 with valid menstrual-regularity response and measured BMI.
3. Use `WTINT2YR` survey weights to form the NHANES BMI-category by irregularity distribution.
4. Create fixed stratified fit/calibration/test partitions (`random_state=42`) before estimating any training weights.
5. Form reweighting ratios from the fit partition only, cap raw weights at five times their positive median, and normalize to mean one. Calibration and test feature distributions remain unseen.
6. Train `HistGradientBoostingClassifier(max_depth=4, max_iter=150)` raw and population-reweighted models; report performance only on the held-out test partition. “Reweighted” does not mean probability-calibrated.

## Current generated run

| Item | Value |
|---|---:|
| Fit / calibration / test rows | 6400 / 1600 / 2000 |
| Held-out raw ROC AUC | 0.6323 |
| Held-out weighted ROC AUC | 0.6422 |
| Weighted ROC AUC 95% bootstrap CI | 0.6199 to 0.6661 |
| Held-out weighted average precision | 0.5248 |
| Held-out weighted Brier score (lower is better) | 0.2308 |
| Held-out weighted expected calibration error (lower is better) | 0.0415 |
| Mean absolute model divergence | 0.0667 |
| Maximum model divergence | 0.4208 |
| Test predictions shifting by more than 0.15 | 9.3% |
| NHANES complete reference respondents | 1206 |
| Raw weights capped | 29.2% |

The weighted model is not expected to improve AUC on a held-out sample drawn from the same labelled population. Its value is to document when predictions are sensitive to a measured mismatch with a reference population. These are internal estimates from labels with unconfirmed provenance, not clinical validation or evidence of diagnostic accuracy. `artifacts/training_summary.json` is the machine-readable source of truth for this run.
