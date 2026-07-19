"""Train reproducible raw and NHANES-calibrated Trace research models."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from model.audit_and_reweight import audit_and_reweight, bmi_category

FEATURES = ["Age", "BMI", "Menstrual_Irregularity", "Chronic_Pain_Level", "Hormone_Level_Abnormality", "Infertility"]
DISCLAIMER = "This is a risk pattern flag, not a diagnosis. It is not medical advice."
SHAP_BACKGROUND_ROWS = 128
BOOTSTRAP_REPLICATES = 1_000


def load_reference(data_dir: Path) -> pd.DataFrame:
    rhq = pd.read_sas(data_dir / "RHQ_L.xpt", format="xport", encoding="latin1")
    demo = pd.read_sas(data_dir / "DEMO_L.xpt", format="xport", encoding="latin1")
    bmx = pd.read_sas(data_dir / "BMX_L.xpt", format="xport", encoding="latin1")
    ref = rhq.merge(demo[["SEQN", "RIAGENDR", "RIDAGEYR", "WTINT2YR"]], on="SEQN").merge(bmx[["SEQN", "BMXBMI"]], on="SEQN")
    ref = ref[(ref.RIAGENDR == 2.0) & ref.RIDAGEYR.between(18, 49) & ref.RHQ031.isin([1.0, 2.0]) & ref.BMXBMI.notna()].copy()
    ref["irregular"] = (ref.RHQ031 == 2.0).astype(int)
    ref["bmi_cat"] = ref.BMXBMI.map(bmi_category)
    return ref


def expected_calibration_error(y: pd.Series, probability: np.ndarray, bins: int = 10) -> float:
    """Estimate calibration error with fixed bins; report it alongside Brier loss."""
    observed = np.asarray(y, dtype=float)
    bucket = np.minimum((probability * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        members = bucket == index
        if members.any():
            error += members.mean() * abs(observed[members].mean() - probability[members].mean())
    return float(error)


def bootstrap_auc_ci(y: pd.Series, probability: np.ndarray, seed: int = 42) -> list[float]:
    """Stratified percentile confidence interval for the held-out ROC AUC."""
    labels = np.asarray(y)
    scores = np.asarray(probability)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if len(class_indices) != 2 or any(len(index) == 0 for index in class_indices):
        raise ValueError("ROC AUC confidence intervals require both outcome classes.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_indices = np.concatenate([rng.choice(index, len(index), replace=True) for index in class_indices])
        values.append(roc_auc_score(labels[sampled_indices], scores[sampled_indices]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def metrics(y: pd.Series, raw: np.ndarray, corrected: np.ndarray) -> dict[str, float | list[float]]:
    diff = np.abs(raw - corrected)
    return {"raw_auc": float(roc_auc_score(y, raw)), "corrected_auc": float(roc_auc_score(y, corrected)),
            "corrected_auc_95_ci": bootstrap_auc_ci(y, corrected),
            "corrected_average_precision": float(average_precision_score(y, corrected)),
            "corrected_brier_score": float(brier_score_loss(y, corrected)),
            "corrected_log_loss": float(log_loss(y, corrected, labels=[0, 1])),
            "corrected_expected_calibration_error": expected_calibration_error(y, corrected),
            "mean_divergence": float(diff.mean()), "max_divergence": float(diff.max()),
            "fraction_shifted_over_015": float((diff > 0.15).mean())}


def explanation_reference(frame: pd.DataFrame) -> dict[str, float]:
    """Persist a training-derived reference point for sensitivity explanations."""
    return {feature: float(frame[feature].median()) for feature in FEATURES}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_markdown(summary: dict) -> str:
    """Render the human-readable benchmark from the saved run summary."""
    metrics_summary = summary["metrics"]
    audit = summary["audit"]
    split = summary["reproducibility"]["split"]
    return f"""# Benchmark methodology

## Scope

This is a research benchmark, not evidence of diagnostic performance. The labelled `structured_endometriosis_data.csv` has unconfirmed provenance and is the only training-label source.

## Reproducible procedure

1. Create standard BMI categories in the labelled data and NHANES reference data.
2. Filter NHANES to women aged 18-49 with valid menstrual-regularity response and measured BMI.
3. Use `WTINT2YR` survey weights to form the NHANES BMI-category by irregularity distribution.
4. Form calibration ratios relative to the labelled data, cap raw weights at five times their positive median, normalize to mean one, and train a weighted model.
5. Use fixed stratified fit/calibration/test partitions (`random_state=42`); the calibration partition is not used to fit either model.
6. Train `HistGradientBoostingClassifier(max_depth=4, max_iter=150)` raw and weighted models; report performance only on the held-out test partition.

## Current generated run

| Item | Value |
|---|---:|
| Fit / calibration / test rows | {split['fit']} / {split['calibration']} / {split['test']} |
| Held-out raw ROC AUC | {metrics_summary['raw_auc']:.4f} |
| Held-out weighted ROC AUC | {metrics_summary['corrected_auc']:.4f} |
| Weighted ROC AUC 95% bootstrap CI | {metrics_summary['corrected_auc_95_ci'][0]:.4f} to {metrics_summary['corrected_auc_95_ci'][1]:.4f} |
| Held-out weighted average precision | {metrics_summary['corrected_average_precision']:.4f} |
| Held-out weighted Brier score (lower is better) | {metrics_summary['corrected_brier_score']:.4f} |
| Held-out weighted expected calibration error (lower is better) | {metrics_summary['corrected_expected_calibration_error']:.4f} |
| Mean absolute model divergence | {metrics_summary['mean_divergence']:.4f} |
| Maximum model divergence | {metrics_summary['max_divergence']:.4f} |
| Test predictions shifting by more than 0.15 | {metrics_summary['fraction_shifted_over_015']:.1%} |
| NHANES complete reference respondents | {summary['reference_rows']} |
| Raw weights capped | {audit['capped_fraction']:.1%} |

The weighted model is not expected to improve AUC on a held-out sample drawn from the same labelled population. Its value is to document when predictions are sensitive to a measured mismatch with a reference population. These are internal estimates from labels with unconfirmed provenance, not clinical validation or evidence of diagnostic accuracy. `artifacts/training_summary.json` is the machine-readable source of truth for this run.
"""


def train(data_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(data_dir / "structured_endometriosis_data.csv")
    source["bmi_cat"] = source.BMI.map(bmi_category)
    reference = load_reference(data_dir)
    weighted, audit = audit_and_reweight(source, reference, "WTINT2YR", [("bmi_cat", "bmi_cat"), ("Menstrual_Irregularity", "irregular")])
    indices = np.arange(len(weighted))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=weighted.Diagnosis, random_state=42)
    fit_idx, calibration_idx = train_test_split(train_idx, test_size=0.2, stratify=weighted.loc[train_idx, "Diagnosis"], random_state=42)
    np.savez(artifact_dir / "split_indices.npz", fit=fit_idx, calibration=calibration_idx, test=test_idx)
    x_fit, x_test = weighted.loc[fit_idx, FEATURES], weighted.loc[test_idx, FEATURES]
    y_fit, y_test = weighted.loc[fit_idx, "Diagnosis"], weighted.loc[test_idx, "Diagnosis"]
    params = dict(random_state=42, max_depth=4, max_iter=150)
    raw_model = HistGradientBoostingClassifier(**params).fit(x_fit, y_fit)
    corrected_model = HistGradientBoostingClassifier(**params).fit(x_fit, y_fit, sample_weight=weighted.loc[fit_idx, "cal_weight"])
    raw = raw_model.predict_proba(x_test)[:, 1]
    corrected = corrected_model.predict_proba(x_test)[:, 1]
    residuals = np.abs(weighted.loc[calibration_idx, "Diagnosis"] - corrected_model.predict_proba(weighted.loc[calibration_idx, FEATURES])[:, 1])
    interval_radius = float(np.quantile(residuals, 0.90, method="higher"))
    joblib.dump(raw_model, artifact_dir / "model_a.joblib")
    joblib.dump(corrected_model, artifact_dir / "model_b.joblib")
    shap_background = weighted.loc[fit_idx, FEATURES].sample(n=min(SHAP_BACKGROUND_ROWS, len(fit_idx)), random_state=42)
    shap_background.to_csv(artifact_dir / "shap_background.csv", index=False)
    summary = {"schema_version": 3, "project_name": "Trace", "features": FEATURES, "disclaimer": DISCLAIMER,
               "validation_status": {"label_provenance_verified": False, "external_clinical_validation": False,
                                     "deployment_status": "research_only_not_clinically_validated"},
               "research_interval_radius": interval_radius, "research_interval_coverage": 0.90,
               "explanation_reference": explanation_reference(weighted.loc[fit_idx]),
               "reproducibility": {"random_seed": 42, "split": {"fit": int(len(fit_idx)), "calibration": int(len(calibration_idx)), "test": int(len(test_idx))},
                                   "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "input_sha256": file_sha256(data_dir / "structured_endometriosis_data.csv")},
               "explainability": {"method": "SHAP permutation explainer", "background_rows": int(len(shap_background)), "background_artifact": "shap_background.csv"},
               "reference_rows": int(len(reference)), "audit": {"raw_weight_min": audit.raw_weight_min, "raw_weight_max": audit.raw_weight_max, "cap": audit.cap, "capped_fraction": audit.capped_fraction},
               "metrics": metrics(y_test, raw, corrected)}
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (artifact_dir.parent / "BENCHMARK.md").write_text(benchmark_markdown(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.artifact_dir), indent=2))
