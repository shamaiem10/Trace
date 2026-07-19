"""Dataset-agnostic calibration weighting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditResult:
    findings: pd.DataFrame
    raw_weight_min: float
    raw_weight_max: float
    cap: float
    capped_fraction: float


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"


def audit_and_reweight(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_weight_col: str | None,
    match_columns: Iterable[tuple[str, str]],
    cap_multiple: float = 5.0,
) -> tuple[pd.DataFrame, AuditResult]:
    """Add stable post-stratification weights without changing row count.

    ``match_columns`` pairs source and reference categorical columns. Empty cells in
    the source are assigned zero raw weight, which makes coverage gaps explicit.
    """
    pairs = list(match_columns)
    if not pairs:
        raise ValueError("At least one source/reference match column pair is required.")
    if df.empty or reference_df.empty:
        raise ValueError("Source and reference data must both contain rows.")
    if cap_multiple <= 0:
        raise ValueError("cap_multiple must be positive.")
    source_cols, reference_cols = zip(*pairs, strict=True)
    result = df.copy()
    source_key = list(source_cols)
    reference_key = list(reference_cols)

    source_dist = result.groupby(source_key, dropna=False).size().rename("source_n")
    source_dist = source_dist / len(result)
    if reference_weight_col is None:
        ref_weights = pd.Series(1.0, index=reference_df.index)
    else:
        ref_weights = reference_df[reference_weight_col].astype(float)
    ref_table = reference_df.assign(_weight=ref_weights).groupby(reference_key, dropna=False)["_weight"].sum()
    ref_dist = ref_table / ref_table.sum()
    # Names differ across source/reference frames, but their ordered cell values
    # describe the same strata. Align by position rather than column label.
    ref_dist.index = ref_dist.index.set_names(source_key)

    aligned = pd.concat([source_dist, ref_dist], axis=1).fillna(0.0)
    aligned.columns = ["source_rate", "reference_rate"]
    aligned["ratio"] = np.where(aligned.source_rate > 0, aligned.reference_rate / aligned.source_rate, np.nan)

    lookup = aligned["ratio"].to_dict()
    def cell_key(row: tuple[object, ...]) -> object:
        return row[0] if len(source_key) == 1 else row

    result["cal_weight_raw"] = [lookup.get(cell_key(row), 0.0) for row in result[source_key].itertuples(index=False, name=None)]
    positive = result.loc[result.cal_weight_raw > 0, "cal_weight_raw"]
    if positive.empty:
        raise ValueError("Reference and source have no overlapping post-stratification cells.")
    cap = cap_multiple * float(positive.median())
    result["cal_weight"] = result.cal_weight_raw.clip(upper=cap)
    result["cal_weight"] /= result.cal_weight.mean()
    audit = AuditResult(
        findings=aligned.reset_index(), raw_weight_min=float(positive.min()), raw_weight_max=float(positive.max()),
        cap=cap, capped_fraction=float((result.cal_weight_raw > cap).mean()),
    )
    return result, audit
