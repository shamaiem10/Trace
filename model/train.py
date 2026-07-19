"""Train reproducible raw and NHANES-calibrated TrustCheck models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model.audit_and_reweight import audit_and_reweight, bmi_category

FEATURES = ["Age", "BMI", "Menstrual_Irregularity", "Chronic_Pain_Level", "Hormone_Level_Abnormality", "Infertility"]
DISCLAIMER = "This is a risk pattern flag, not a diagnosis. It is not medical advice."


def load_reference(data_dir: Path) -> pd.DataFrame:
    rhq = pd.read_sas(data_dir / "RHQ_L.xpt", format="xport", encoding="latin1")
    demo = pd.read_sas(data_dir / "DEMO_L.xpt", format="xport", encoding="latin1")
    bmx = pd.read_sas(data_dir / "BMX_L.xpt", format="xport", encoding="latin1")
    ref = rhq.merge(demo[["SEQN", "RIAGENDR", "RIDAGEYR", "WTINT2YR"]], on="SEQN").merge(bmx[["SEQN", "BMXBMI"]], on="SEQN")
    ref = ref[(ref.RIAGENDR == 2.0) & ref.RIDAGEYR.between(18, 49) & ref.RHQ031.isin([1.0, 2.0]) & ref.BMXBMI.notna()].copy()
    ref["irregular"] = (ref.RHQ031 == 2.0).astype(int)
    ref["bmi_cat"] = ref.BMXBMI.map(bmi_category)
    return ref


def metrics(y: pd.Series, raw: np.ndarray, corrected: np.ndarray) -> dict[str, float]:
    diff = np.abs(raw - corrected)
    return {"raw_auc": float(roc_auc_score(y, raw)), "corrected_auc": float(roc_auc_score(y, corrected)),
            "mean_divergence": float(diff.mean()), "max_divergence": float(diff.max()),
            "fraction_shifted_over_015": float((diff > 0.15).mean())}


def train(data_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(data_dir / "structured_endometriosis_data.csv")
    source["bmi_cat"] = source.BMI.map(bmi_category)
    reference = load_reference(data_dir)
    weighted, audit = audit_and_reweight(source, reference, "WTINT2YR", [("bmi_cat", "bmi_cat"), ("Menstrual_Irregularity", "irregular")])
    indices = np.arange(len(weighted))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=weighted.Diagnosis, random_state=42)
    np.savez(artifact_dir / "split_indices.npz", train=train_idx, test=test_idx)
    x_train, x_test = weighted.loc[train_idx, FEATURES], weighted.loc[test_idx, FEATURES]
    y_train, y_test = weighted.loc[train_idx, "Diagnosis"], weighted.loc[test_idx, "Diagnosis"]
    params = dict(random_state=42, max_depth=4, max_iter=150)
    raw_model = HistGradientBoostingClassifier(**params).fit(x_train, y_train)
    corrected_model = HistGradientBoostingClassifier(**params).fit(x_train, y_train, sample_weight=weighted.loc[train_idx, "cal_weight"])
    raw = raw_model.predict_proba(x_test)[:, 1]
    corrected = corrected_model.predict_proba(x_test)[:, 1]
    calibration_idx, _ = train_test_split(train_idx, test_size=0.75, stratify=weighted.loc[train_idx, "Diagnosis"], random_state=42)
    residuals = np.abs(weighted.loc[calibration_idx, "Diagnosis"] - corrected_model.predict_proba(weighted.loc[calibration_idx, FEATURES])[:, 1])
    interval_radius = float(np.quantile(residuals, 0.90, method="higher"))
    joblib.dump(raw_model, artifact_dir / "model_a.joblib")
    joblib.dump(corrected_model, artifact_dir / "model_b.joblib")
    summary = {"features": FEATURES, "disclaimer": DISCLAIMER, "interval_radius": interval_radius,
               "reference_rows": int(len(reference)), "audit": {"raw_weight_min": audit.raw_weight_min, "raw_weight_max": audit.raw_weight_max, "cap": audit.cap, "capped_fraction": audit.capped_fraction},
               "metrics": metrics(y_test, raw, corrected)}
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.artifact_dir), indent=2))
