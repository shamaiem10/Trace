# Trace

### The missing research layer between a symptom log and trustworthy AI for women's health.

![Trace interface artwork](app/assets/traceimage.jpeg)

**Trace turns fragmented, one-off observations into a reusable research asset:** a privacy-bounded longitudinal record, a reproducible calibration-aware model, and an explainable API that makes uncertainty visible rather than hiding it behind a score.

> **Research infrastructure - not a diagnostic product.** Trace is deliberately non-clinical. It does not diagnose, predict treatment, or offer medical advice.

---

## The investment thesis

Women's hormonal health has a data-compounding problem, not just a user-interface problem. Symptoms vary with time, sleep, stress, and cycle context; research data are fragmented; and a model trained on a convenient dataset can look credible while silently failing to represent the population it hopes to serve.

Trace is built to make that problem tractable. It contributes a Layer 2 model infrastructure asset with a Layer 3 application surface: researchers can reproduce the model and audit its assumptions, while participants can contribute structured, de-identified longitudinal observations that are ready for future benchmarks.

**The opportunity:** do not make another opaque women's-health score. Build the accountable data and model layer that future products, studies, and clinical research workflows can extend.

## What Trace solves that a typical health demo leaves behind

| Research pain point | What usually happens | What Trace implements |
| --- | --- | --- |
| **Dynamic biology, static snapshots** | A single appointment or form collapses a changing experience into one row. | An open daily-trajectory contract for cycle context, flow, pain, fatigue, migraine, brain fog, sleep, and stress. Missing values remain visible rather than being fabricated. |
| **Population mismatch is invisible** | Models are trained on whatever data is available, then their population fit is assumed. | A reproducible post-stratification audit against survey-weighted NHANES reference data, plus a raw-versus-calibrated model comparison. |
| **Model outputs are opaque** | A probability is presented as a conclusion. | Local and global SHAP-based model-sensitivity explanations, an explicit model-agreement signal, and a research prediction range. |
| **Reproducibility ends at the slide deck** | Splits, packages, preprocessing choices, and evaluation assumptions are hard to recover. | Fixed split indices, source-data hash, pinned package versions, saved model artifacts, training code, benchmark methodology, and API tests. |
| **Data quality gets hidden** | Blank fields are imputed or ignored, making downstream findings look cleaner than the evidence. | Per-measure missingness, observed counts, descriptive ranges, and trend estimates only when repeated observations exist. |
| **Privacy is an afterthought** | Free text or raw records travel to an AI endpoint without a hard boundary. | Identifier-free schema, strict rejection of extra fields, and an opt-in AI reflection that accepts only aggregated summaries - never dates or raw daily observations. |
| **Polished apps overstate certainty** | A visually polished tracker makes unsupported health claims. | Non-diagnostic language and validation status are part of the API contract, model metadata, tests, and user experience. |

## Why Layer 2 is the core of Trace

The Hack-Nation challenge asks teams to leave behind a reusable model that future researchers can reproduce, extend, and combine. That is Trace's center of gravity.

```text
Longitudinal record layer
        |
        v
Open schema + quality signals
        |
        v
Population-reality audit -> calibration-aware research model -> explainability API
        |
        v
Reusable artifacts, benchmark methodology, tests, and a safe application demo
```

### A model that exposes its own limits

Trace trains raw and calibration-weighted `HistGradientBoostingClassifier` models using fixed, stratified fit / calibration / test partitions. The weighted model uses post-stratification across BMI category x menstrual-irregularity, referenced to survey-weighted NHANES data. Weight capping prevents a small number of observations from dominating training.

The purpose is not to manufacture a clinical claim or chase a leaderboard. It is to show when a research score is sensitive to a documented dataset mismatch.

| Current reproducible run | Result |
| --- | ---: |
| Fit / calibration / test rows | 6,400 / 1,600 / 2,000 |
| NHANES reference respondents | 1,206 |
| Held-out weighted ROC AUC | 0.6412 |
| Held-out weighted calibration error | 0.0388 |
| Test predictions shifted by more than 0.15 after weighting | 8.5% |
| Rows whose raw weights were capped | 29.0% |

These are internal, held-out research metrics from labels with unconfirmed provenance. They are **not** clinical validation or evidence of diagnostic accuracy. The full methodology, including the limitations, is in [BENCHMARK.md](BENCHMARK.md) and [the data-debt report](data/DATA_DEBT_REPORT.md).

## The Layer 3 proof: a contribution path, not a data silo

Trace's interface is intentionally a demonstration of how a future data-collection experience can respect research constraints:

- Daily entries are structured, de-identified, and validated at the API boundary.
- Dates must be unique, at least two observation days are required, and every day must include a measurement.
- The UI returns descriptive patterns and data completeness - not disease labels.
- The optional AI reflection is explicitly consented to and receives only an aggregate summary.

This means the app is not an isolated tracker. It is a usable front door to a durable, benchmark-ready record format.

## Built as open scientific infrastructure

| Reusable asset | Location |
| --- | --- |
| Longitudinal record specification | [data/TRACE_TRAJECTORY_SCHEMA.md](data/TRACE_TRAJECTORY_SCHEMA.md) |
| Benchmark methodology and model limits | [BENCHMARK.md](BENCHMARK.md) |
| Dataset mismatch audit | [data/DATA_DEBT_REPORT.md](data/DATA_DEBT_REPORT.md) |
| Training and calibration pipeline | [model/train.py](model/train.py) |
| Reweighting utility | [model/audit_and_reweight.py](model/audit_and_reweight.py) |
| Saved models, splits, background sample, and run metadata | [artifacts/](artifacts/) |
| FastAPI contract and explainability endpoints | [api/main.py](api/main.py) |
| Safety and reproducibility tests | [tests/](tests/) |

### API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/trajectory/schema` | Publishes the identifier-free daily-record contract. |
| `POST /v1/trajectory` | Validates observations and returns descriptive patterns plus data-quality signals. |
| `POST /v1/trajectory/explain` | Returns an opt-in aggregate-only AI research reflection. |
| `POST /v1/assess` | Returns raw and calibrated research scores, agreement, uncertainty, and local factors. |
| `GET /v1/explain/global` | Returns global SHAP sensitivity information. |
| `GET /v1/data-debt-report` | Exposes the documented distributional audit. |

## A credible path to compounding value

1. **Standardize contribution.** Make the Trace trajectory schema a dependable unit for prospective longitudinal collection.
2. **Add modalities responsibly.** Expand with consented wearables, hormone measurements, laboratory data, and validated labels; publish source-specific cards and licensing terms.
3. **Benchmark by person, not by day.** Release person-level splits, missing-data policy, subgroup evaluation, and externally testable tasks.
4. **Continuously audit representation.** Re-run the population-reality audit as sources change, report coverage gaps, and keep model cards current.
5. **Earn clinical relevance rather than implying it.** Pursue prospective validation, governance, and regulatory review before any care-facing use.

## Run Trace locally

```powershell
py -3.11 -m venv .trace-env
.\.trace-env\Scripts\python.exe -m pip install --upgrade pip
.\.trace-env\Scripts\python.exe -m pip install -r requirements.txt
.\.trace-env\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the app and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

To run the checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Optional AI reflection

Copy `.env.example` to `.env` and provide `OPENAI_API_KEY`. This feature is optional: it receives only an already-aggregated research summary, not raw records or dates. `TRACE_OPENAI_MODEL` may override the default model.

## Responsible scope

Trace does **not** solve the full women's-health data gap today. Its labelled source has unconfirmed provenance, it has no external clinical validation, and it should not inform clinical decisions. Those are surfaced directly in the code and artifacts because accountable infrastructure begins by making its limits inspectable.

## License

Code is MIT licensed. Project documentation is CC-BY-4.0. Original source datasets retain their own terms.
