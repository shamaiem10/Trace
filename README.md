# Trace

> **A gentle, research-first space for making longitudinal health observations more legible.**

![Trace watercolor hero](app/assets/traceimage.jpeg)

Trace is an open research prototype for menstrual-health infrastructure. Its lilac-and-blue interface keeps daily records calm and approachable while its API makes data quality, missingness, and uncertainty explicit.

> [!WARNING]
> **Not a diagnostic tool or medical device.** Trace produces research summaries only; it does not diagnose, predict treatment, or provide medical advice.

## Why Trace?

| Lilac clarity | Blue transparency | Research boundaries |
| --- | --- | --- |
| A soft, accessible interface for daily observations | Completeness and missingness stay visible | Identifier-free records and non-clinical language by design |

## What it includes

- A responsive lilac-and-blue browser experience for entering daily symptoms and cycle context.
- A reusable, de-identified daily-trajectory schema with strict validation.
- Descriptive symptom patterns, completeness indicators, and an exact API response view.
- An optional, aggregate-only AI reflection that excludes dates and raw daily observations.
- Calibration-weighted baseline research scoring, model-sensitivity explanations, and a population-reality audit.

## Project map

```text
app/                  Front-end experience and visual assets
api/main.py           FastAPI service and research-only endpoints
data/                 Schema, datasheets, source data, and debt report
model/                Training and population-audit workflows
artifacts/            Reproducible model outputs and metadata
tests/                API and audit tests
```

## Run locally

Trace runs as a single FastAPI app. Use the pinned dependencies because the saved artifacts and their dependency versions are a matched set.

```powershell
py -3.11 -m venv .trace-env
.\.trace-env\Scripts\python.exe -m pip install --upgrade pip
.\.trace-env\Scripts\python.exe -m pip install -r requirements.txt
.\.trace-env\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to use the app, or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API documentation.

### Optional AI reflection

Copy `.env.example` to `.env`, then add an `OPENAI_API_KEY`. The explanation is opt-in and Trace only sends an already-aggregated summary—never dates or raw observations. `TRACE_OPENAI_MODEL` can override the default model.

## API at a glance

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/trajectory/schema` | Returns the portable, identifier-free record contract. |
| `POST /v1/trajectory` | Validates observations and returns descriptive summaries with data quality. |
| `POST /v1/trajectory/explain` | Generates an opt-in aggregate-only research reflection. |
| `POST /v1/assess` | Returns a calibration-weighted baseline research score and local factors. |
| `GET /v1/explain/global` | Returns global model sensitivity information. |
| `GET /v1/data-debt-report` | Returns the documented distributional audit. |

## Reproducibility and limits

Training uses a fixed seed (`42`) and stratified fit, calibration, and test partitions. `artifacts/training_summary.json` records split indices, source-data hashes, dependency versions, model parameters, audit outputs, and held-out evaluation metrics.

The supplied NHANES files are reference-only and are not linked to labelled training data. The labelled dataset has unconfirmed provenance and must not be interpreted as clinical ground truth. Internal model metrics do not establish real-world accuracy.

Before using or extending this work, read [the benchmark notes](BENCHMARK.md), [the trajectory schema](data/TRACE_TRAJECTORY_SCHEMA.md), and [the data-debt report](data/DATA_DEBT_REPORT.md).

## License

Code is MIT licensed. Project documentation is CC-BY-4.0. Original source datasets retain their own terms.
