import pandas as pd

from model.audit_and_reweight import audit_and_reweight, bmi_category
from model.train import benchmark_markdown


def test_audit_adds_normalized_weight_without_losing_rows():
    source = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    reference = pd.DataFrame({"reference_group": ["a", "b", "b"], "weight": [1.0, 1.0, 2.0]})
    result, audit = audit_and_reweight(source, reference, "weight", [("group", "reference_group")])
    assert len(result) == len(source)
    assert set(["cal_weight_raw", "cal_weight"]).issubset(result)
    assert round(result.cal_weight.mean(), 10) == 1.0
    assert audit.capped_fraction >= 0


def test_bmi_categories_handle_boundaries():
    assert bmi_category(18.4) == "Underweight"
    assert bmi_category(18.5) == "Normal"
    assert bmi_category(25) == "Overweight"
    assert bmi_category(30) == "Obese"


def test_benchmark_is_rendered_from_summary():
    summary = {"metrics": {"raw_auc": 0.6, "corrected_auc": 0.61, "corrected_auc_95_ci": [0.5, 0.7], "corrected_average_precision": 0.55, "corrected_brier_score": 0.2, "corrected_expected_calibration_error": 0.1, "mean_divergence": 0.1, "max_divergence": 0.2, "fraction_shifted_over_015": 0.03}, "audit": {"capped_fraction": 0.2}, "reproducibility": {"split": {"fit": 64, "calibration": 16, "test": 20}}, "reference_rows": 12}
    rendered = benchmark_markdown(summary)
    assert "0.6100" in rendered
    assert "64 / 16 / 20" in rendered
