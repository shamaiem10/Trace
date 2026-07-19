# TrustCheck

TrustCheck is a research prototype that pairs an endometriosis risk-pattern score with a per-prediction agreement signal, transparent feature impacts, and a population-reality audit.

> **Not a diagnostic tool.** Every result is a risk-pattern flag, not a diagnosis or medical advice.

## What is included

- `model/audit_and_reweight.py` — reusable calibration-weighting audit.
- `model/train.py` — deterministic training of raw and NHANES-calibrated models.
- `api/main.py` — FastAPI service: assessment, counterfactual, and data-debt endpoints.
- `app/index.html` — small browser demo that calls the API (a handoff-ready replacement for a Lovable export).
- `data/` — benchmark, data debt report, and datasheets.
- `artifacts/` — trained models, exact train/test indices, and run summary.

## Run locally

Use Python 3.12+.

```powershell
python -m pip install -r requirements.txt
python -m model.train
python -m uvicorn api.main:app --reload
```

Open `app/index.html` in a browser after starting the API, or use the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Reproduce the model

The four provided data files are copied to `data/raw/`. Run `python -m model.train`; the seed (`42`), 80/20 stratified split, model parameters, capped weights, serialized models, and split indices are all saved. The NHANES files are reference-only and are never patient-level merged with the labelled dataset.

## API contract

- `POST /v1/assess` accepts six features in snake case and returns the corrected risk-pattern score, raw score, Trust Score, broad research calibration range, three transparent factors, and disclaimer.
- `POST /v1/counterfactual` rescoring endpoint; it changes exactly one selected feature.
- `GET /v1/data-debt-report` returns the audit report.

See [BENCHMARK.md](data/BENCHMARK.md) and [DATA_DEBT_REPORT.md](data/DATA_DEBT_REPORT.md) before interpreting any model result.

## License

Code is MIT licensed. Project documentation is CC-BY-4.0. Original source datasets retain their own terms; the structured labelled dataset has unconfirmed provenance.
