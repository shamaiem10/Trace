# Trace trajectory record schema

Version: 1.0.0

## Purpose

This open, identifier-free schema represents one person's self-reported daily menstrual-health observations. It is designed for research interoperability, longitudinal data-quality assessment, and benchmark construction. It is not a clinical record, a diagnostic instrument, or a substitute for care.

## Unit of observation

One row represents one calendar day. `observed_on` must be unique within a record. Dates should use ISO 8601 (`YYYY-MM-DD`). A record must contain at least two days and no more than 120 days per API submission.

| Field | Type | Range | Required | Meaning |
|---|---|---:|---|---|
| `observed_on` | date | ISO 8601 | Yes | Calendar date of the observation. |
| `cycle_day` | integer | 1–60 | No | Self-reported day of the menstrual cycle. |
| `bleeding_flow` | integer | 0–4 | No | Self-reported ordinal bleeding-flow level. |
| `pain_level` | number | 0–10 | No | Self-reported pain level. |
| `fatigue_level` | number | 0–10 | No | Self-reported fatigue level. |
| `migraine_level` | number | 0–10 | No | Self-reported migraine level. |
| `brain_fog_level` | number | 0–10 | No | Self-reported brain-fog level. |
| `sleep_hours` | number | 0–24 | No | Self-reported sleep duration. |
| `stress_level` | number | 0–10 | No | Self-reported stress level. |

Each row must contain at least one measure besides its date. Blank values mean not observed; they must remain blank rather than being inferred or imputed in the source record.

## Privacy and governance

Never include direct identifiers (for example name, email, phone number, address, exact location, medical-record number, device ID, or free-text clinical notes). Trace rejects unrecognized fields at the API boundary. Before sharing research data, apply the applicable consent, ethics review, privacy, licensing, and data-governance process.

## Interpretation

`POST /v1/trajectory` returns the original record, missingness per measure, descriptive ranges/means, and an optional linear change-per-week estimate. The estimate describes the submitted values only; it is not evidence of a biological trend, ovulation, disease, diagnosis, prognosis, or treatment effect.

## Benchmark extension guidance

Future benchmark releases should publish source-specific data cards, a fixed train/validation/test split by person (never by day), label provenance, missing-data policy, consent/licence terms, and a subgroup/equity evaluation plan. Do not combine different datasets into person-level records unless the linkage is legitimate and documented.
