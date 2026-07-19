import pandas as pd

from model.audit_and_reweight import audit_and_reweight


def test_audit_adds_normalized_weight_without_losing_rows():
    source = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    reference = pd.DataFrame({"reference_group": ["a", "b", "b"], "weight": [1.0, 1.0, 2.0]})
    result, audit = audit_and_reweight(source, reference, "weight", [("group", "reference_group")])
    assert len(result) == len(source)
    assert set(["cal_weight_raw", "cal_weight"]).issubset(result)
    assert round(result.cal_weight.mean(), 10) == 1.0
    assert audit.capped_fraction >= 0
