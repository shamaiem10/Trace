# Trace

Trace is an open research prototype for menstrual-health infrastructure. It pairs a de-identified longitudinal trajectory schema with data-quality reporting, a calibration-weighted baseline score, model-sensitivity explanations, and a population-reality audit.

> **Not a diagnostic tool.** Every result is a risk-pattern flag, not a diagnosis or medical advice.

## What is included

- `model/audit_and_reweight.py` - reusable post-stratification weighting audit.
- `model/train.py` - deterministic raw and NHANES-calibrated model training, using independent fit, calibration, and test partitions.
- `api/main.py` - FastAPI service with a reusable daily-trajectory schema, trajectory summaries, baseline assessment, SHAP explanations, counterfactual, and data-debt endpoints.
- `app/index.html` - local browser demo for creating a non-diagnostic research trajectory record.
- `data/TRACE_TRAJECTORY_SCHEMA.md` - portable daily-observation schema, privacy boundary, and benchmark-extension rules.
- `data/` - benchmark methodology, data-debt report, and dataset datasheets.
- `artifacts/` - model checkpoints, split indices, and reproducibility metadata.

## Run locally

Use one project-local virtual environment and the exact pinned dependencies. Model artifacts and dependency versions are a matched set; always retrain after changing Python, NumPy, or scikit-learn.

```powershell
py -3.11 -m venv .trace-env
.\.trace-env\Scripts\python.exe -m pip install --upgrade pip
.\.trace-env\Scripts\python.exe -m pip install -r requirements.txt
.\.trace-env\Scripts\python.exe -m model.train
.\.trace-env\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` after starting the API, or use `http://127.0.0.1:8000/docs`.

## Reproduce the model

The four supplied source data files are in `data/raw/`. Training uses a fixed seed (`42`) and stratified fit/calibration/test partitions. It saves the split indices, source-data hash, dependency version, model parameters, capped weighting audit, and held-out discrimination, calibration, and bootstrap-uncertainty metrics in `artifacts/training_summary.json`.

The NHANES files are reference-only: they create a survey-weighted BMI-category by menstrual-irregularity distribution, but no NHANES record is linked to the labelled training data. The labelled dataset has unconfirmed provenance and must not be treated as clinical ground truth. Trace is therefore explicitly marked `research_only_not_clinically_validated`; its internal metrics do not establish real-world accuracy.

## API contract

- `GET /v1/trajectory/schema` publishes Trace's identifier-free daily-observation contract.
- `POST /v1/trajectory` validates daily observations and returns the original research record, data completeness, cycle context, and descriptive trends.
- `POST /v1/assess` returns a calibration-weighted research score, raw-model score, model-agreement score, held-out research prediction range, and local SHAP factors.
- `POST /v1/counterfactual` changes exactly one chosen input and re-scores the weighted model.
- `GET /v1/explain/global` returns global feature importance as mean absolute SHAP values over the saved background sample.
- `GET /v1/data-debt-report` returns the documented distributional audit.

The trajectory endpoint is the project's primary infrastructure asset: it deliberately accepts no identifiers and produces no diagnosis. `model_agreement_score` describes agreement between the raw and calibration-weighted research models; it is not clinical certainty, validation, or a diagnosis. SHAP values show how inputs move the model score relative to its saved background dataset; they are not causal or medical explanations.

See [BENCHMARK.md](BENCHMARK.md) and [DATA_DEBT_REPORT.md](data/DATA_DEBT_REPORT.md) before interpreting any model result.

## License

Code is MIT licensed. Project documentation is CC-BY-4.0. Original source datasets retain their own terms; the structured labelled dataset has unconfirmed provenance.
