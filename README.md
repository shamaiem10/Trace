<div align="center">

<img src="app/assets/traceimage.jpeg" alt="Trace watercolor research artwork" width="100%" />

# Trace

### Audit before you claim.

**An open, reproducible model-audit layer for women’s hormonal-health research.**

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge)](#contributing)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
<br>
*A submission for Hack-Nation Challenge 05: Foundation Models for Women’s Hormonal Health.*

</div>

> [!IMPORTANT]
> **Trace is research infrastructure, not a medical device.** It does not diagnose endometriosis, determine whether a person has a condition, recommend treatment, or provide medical advice. The labelled source has unverified provenance and the model has no external clinical validation.

---

## The idea in one minute

Many health-AI demos begin with a dataset, train one model, and present a polished score. That skips the question researchers must answer first: **does the available training data resemble the population the model is meant to represent?**

Trace turns that question into a reproducible workflow:

```text
Labelled research dataset                     NHANES public survey files
(features + unverified target)                 (population reference only)
              │                                           │
              └──────────── representation audit ─────────┘
                                      │
                         fit-only population weights
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              Model A: raw                       Model B: reweighted
                    │                                   │
                    └──────── same held-out test ───────┘
                                      │
                    metrics + uncertainty + SHAP + limits
                                      │
                         researcher-facing audit studio
```

The same fictional profile can be sent through both already-trained models. Trace returns both research scores, their difference, an uncertainty band, and a local explanation of Model B. **No retraining happens during this request, and the sample is not stored.**

---

## Why Trace exists

Women’s hormonal health is dynamic, underrepresented, and difficult to reduce to a single snapshot. Research data are often fragmented across studies, feature definitions, collection schedules, and institutions. Even when a model is technically valid, it can silently inherit the recruitment patterns of a convenient dataset.

Trace contributes the reusable layer requested by the Hack-Nation brief:

| Challenge need | Trace contribution |
| --- | --- |
| Reusable AI model infrastructure | Two saved models, deterministic splits, training code, evaluation artifacts, and explicit package versions |
| Scientific rigor | A fit/calibration/test design that estimates reweighting factors from the fit partition only |
| Explainability | Local and global permutation SHAP explanations in model-score units |
| Open research documentation | Model card API, generated benchmark, data-debt report, dataset datasheets, and trajectory schema |
| Application infrastructure | A researcher-facing audit studio and strict, identifier-free longitudinal API contract |
| Responsible claims | Research-only language, forbidden extra profile fields, surfaced uncertainty, and validation-status metadata |

Trace is therefore **not another endometriosis prediction app**. It is a quality-control and transparency layer that researchers can reproduce, inspect, extend, and eventually replace with clinically verified data.

---

## What the demonstration shows

The frontend is a live view of the backend artifacts—not a collection of hard-coded slides.

### 1. Data and representation audit

The interface identifies the labelled training source and the NHANES population reference separately. It aggregates the eight BMI-category × menstrual-irregularity strata into four readable BMI comparisons:

| BMI category | Labelled source | Survey-weighted NHANES reference |
| --- | ---: | ---: |
| Underweight | 12.8% | 2.7% |
| Normal | 56.1% | 32.3% |
| Overweight | 27.2% | 27.1% |
| Obese | 4.0% | 38.0% |

These values demonstrate a documented **feature-distribution mismatch**. They do not prove bias in every population, validate the target labels, or establish clinical performance.

### 2. Held-out model results

The dashboard reads metrics from the generated `artifacts/training_summary.json` file:

| Current reproducible run | Value |
| --- | ---: |
| Fit / calibration / test rows | 6,400 / 1,600 / 2,000 |
| Raw Model A ROC AUC | 0.6323 |
| Reweighted Model B ROC AUC | 0.6422 |
| Model B ROC AUC 95% bootstrap interval | 0.6199–0.6661 |
| Model B average precision | 0.5248 |
| Model B Brier score | 0.2308 |
| Model B expected calibration error | 0.0415 |
| Mean absolute A/B score divergence | 0.0667 |
| Test scores changing by more than 0.15 | 9.3% |
| Fit rows with capped raw weights | 29.2% |

These are internal held-out estimates from an unverified labelled source. A small metric improvement is not the project’s central claim; the contribution is making population sensitivity measurable and inspectable.

### 3. Synthetic profile sandbox

A researcher can enter six fictional values:

- age;
- BMI;
- menstrual-irregularity indicator;
- chronic-pain level;
- hormone-level-abnormality indicator; and
- infertility indicator.

The backend converts that input into the exact feature order saved with the training artifact and performs inference:

```text
fictional profile ──► saved Model A ──► raw research score
        │
        └───────────► saved Model B ──► reweighted research score
                                              │
                                   permutation SHAP explanation
```

The response includes:

- the raw and population-reweighted scores;
- the absolute A/B score difference;
- a held-out residual-error band;
- the three strongest local SHAP contributions;
- the explanation method and background value;
- artifact and validation metadata; and
- a non-diagnostic disclaimer.

The request performs **inference only**. It does not retrain either model, update the weights, append the profile to a dataset, or persist the submission.

### 4. Global explainability

The global panel requests SHAP values for 32 rows from the saved 128-row background sample and ranks the six features by mean absolute contribution. This communicates how Model B behaves across that background sample. It must not be interpreted as biological importance, causality, or a clinical risk factor ranking.

### 5. Research-readiness and limits

The interface deliberately places positive evidence and unresolved gaps together:

- held-out testing: implemented;
- deterministic split indices: published;
- representation audit: implemented;
- SHAP sensitivity: implemented;
- label provenance verified: **no**;
- external clinical validation: **no**;
- deployment status: **research only**.

---

## Datasets

Trace uses one labelled research table and three public NHANES component files. The files supplied with the project match the files used to create the current artifacts.

### Labelled structured source

**File:** `data/raw/structured_endometriosis_data.csv`

| Property | Value |
| --- | --- |
| Rows | 10,000 |
| Columns | 7 |
| Target | `Diagnosis` |
| Target distribution | 5,921 class `0`; 4,079 class `1` |
| Missing values | None observed |
| Provenance / licence | Not stated in the supplied file; therefore unconfirmed |

The six model features are:

| Feature | Type used by Trace | API range |
| --- | --- | ---: |
| `Age` | integer | 18–49 |
| `BMI` | continuous | 10–80 |
| `Menstrual_Irregularity` | binary indicator | 0 or 1 |
| `Chronic_Pain_Level` | continuous | 0–10 |
| `Hormone_Level_Abnormality` | binary indicator | 0 or 1 |
| `Infertility` | binary indicator | 0 or 1 |

The target is used only for internal model fitting and held-out evaluation. Trace does not claim that it is clinically adjudicated ground truth.

### NHANES reference components

Trace uses the supplied NHANES 2021–2023 component files:

| File | Component | Fields used |
| --- | --- | --- |
| `DEMO_L.xpt` | Demographics | `SEQN`, sex (`RIAGENDR`), age (`RIDAGEYR`), interview survey weight (`WTINT2YR`) |
| `BMX_L.xpt` | Body Measures | respondent key (`SEQN`), measured BMI (`BMXBMI`) |
| `RHQ_L.xpt` | Reproductive Health | respondent key (`SEQN`), menstrual-regularity response (`RHQ031`) |

NHANES is used only to estimate a survey-weighted reference distribution. It does **not** provide endometriosis labels to Trace, its rows are not appended to the labelled dataset, and no person-level linkage is attempted between the two sources.

### How the NHANES files are combined

The training pipeline performs the following deterministic operations:

1. Read each SAS XPORT file with pandas.
2. Select the necessary demographic fields from `DEMO_L.xpt`.
3. Inner-join reproductive health and demographics on the NHANES respondent identifier `SEQN`.
4. Inner-join measured BMI on `SEQN`.
5. Retain respondents coded female (`RIAGENDR == 2`).
6. Retain ages 18–49, matching the research profile boundary.
7. Retain valid `RHQ031` responses `1` or `2` and non-missing measured BMI.
8. Encode `RHQ031 == 2` as the irregularity indicator.
9. Convert measured BMI into standard categories: underweight `<18.5`, normal `<25`, overweight `<30`, and obese `≥30`.
10. Use `WTINT2YR` when calculating the BMI-category × menstrual-irregularity reference distribution.

The resulting complete reference extract contains **1,206 respondents**. This count is not presented as the total NHANES population; survey weights are required for population-rate estimates.

---

## Population audit and Model B

### Why two models?

Model A provides a baseline trained with ordinary equal row influence. Model B asks a controlled research question: *how does model behavior change when the fit data are reweighted toward the selected NHANES feature distribution?*

Both are `HistGradientBoostingClassifier` models with identical parameters:

```python
HistGradientBoostingClassifier(
    random_state=42,
    max_depth=4,
    max_iter=150,
)
```

### Post-stratification procedure

For each BMI-category × irregularity cell, Trace computes:

```text
raw cell weight = NHANES survey-weighted cell rate / fit-partition cell rate
```

Each labelled fit row inherits the weight for its cell. Raw positive weights are capped at five times their positive median, then normalized to mean one. Capping prevents a very sparse source cell from dominating the fitted model.

Crucially, split creation happens **before** weight estimation:

1. create the 80% train / 20% test split with target stratification;
2. divide the train portion into 6,400 fit and 1,600 calibration rows;
3. estimate reweighting ratios from the 6,400 fit rows only;
4. fit Model A and Model B on the same fit rows;
5. use the calibration rows only for the residual-error radius; and
6. calculate reported performance on the untouched 2,000-row test partition.

This prevents the calibration or held-out feature distributions from leaking into Model B’s training weights.

> [!NOTE]
> Model B is **population-reweighted**, not probability-calibrated. Trace reports calibration-oriented metrics such as Brier score and expected calibration error, but it does not claim that post-stratification itself calibrates predicted probabilities.

---

## Evaluation and uncertainty

The pipeline reports:

- ROC AUC for Model A and Model B;
- a stratified 1,000-replicate percentile bootstrap interval for Model B ROC AUC;
- average precision;
- Brier score;
- log loss;
- ten-bin expected calibration error;
- mean and maximum absolute Model A/B divergence; and
- the fraction of test predictions whose absolute divergence exceeds 0.15.

For the sandbox uncertainty display, Model B scores the separate 1,600-row calibration partition. Trace computes the absolute residual `|label - score|` and saves the higher-method 90th percentile as a global radius. The API clips `score ± radius` to `[0, 1]`.

This is intentionally presented as a **broad research error band**, not an individual confidence interval, prognosis, or guarantee. A future clinically oriented system would require stronger uncertainty methodology, verified labels, subgroup validation, and prospective evaluation.

---

## Explainability with SHAP

Trace saves a deterministic 128-row sample from the fit partition as `artifacts/shap_background.csv`.

The API wraps Model B’s positive-class `predict_proba` output and creates a permutation `shap.Explainer`:

- **Local explanation:** evaluates one fictional profile with `2 × features + 1` model evaluations and returns the three largest absolute SHAP values.
- **Global explanation:** evaluates the first 32 saved background rows and ranks features by mean absolute SHAP value.
- **Dependency fallback:** if SHAP is unavailable, local inspection falls back to a clearly labelled baseline-sensitivity calculation; the global SHAP endpoint returns an explicit `503` instead of fabricating an explanation.

SHAP values describe sensitivity of the fitted model around its background distribution. They do not explain physiology, establish causality, or validate the underlying labels.

---

## Longitudinal research layer

The project also publishes an identifier-free daily trajectory contract for future research collection. A record can contain 2–120 unique dates with optional:

- cycle day;
- bleeding flow;
- pain;
- fatigue;
- migraine;
- brain fog;
- sleep duration; and
- stress.

The API preserves raw missing values and returns observed counts, missing rates, ranges, means, and a descriptive linear change per week when repeated measurements exist. Unknown fields are rejected, which prevents names, emails, free-text notes, and other unapproved identifiers from entering the contract.

An optional OpenAI Responses API endpoint accepts only an already-aggregated summary. Raw daily records and dates are rejected before any provider request. Its prompt is bounded to descriptive, non-diagnostic language.

See [the full trajectory specification](data/TRACE_TRAJECTORY_SCHEMA.md).

---

## Architecture

```text
┌──────────────────────────── Browser audit studio ────────────────────────────┐
│ Data audit │ held-out metrics │ synthetic inspection │ SHAP │ limitations   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ JSON over HTTP
┌──────────────────────────────────────▼───────────────────────────────────────┐
│ FastAPI                                                                   │
│ strict Pydantic contracts │ artifact checks │ trajectory summary │ AI guard │
└──────────────┬────────────────────────┬───────────────────────────┬───────────┘
               │                        │                           │
        Model A + Model B        training_summary.json       SHAP background
               ▲                        ▲                           ▲
               └────────────────────────┴───────────────────────────┘
                                        │ generated by
┌───────────────────────────────────────▼──────────────────────────────────────┐
│ Reproducible training pipeline                                              │
│ fixed split → fit-only audit → raw/reweighted fit → calibration → test      │
└───────────────────┬───────────────────────────────────┬──────────────────────┘
                    │                                   │
       labelled structured CSV             joined, filtered, weighted NHANES
```

### Repository map

| Path | Purpose |
| --- | --- |
| `api/main.py` | FastAPI service, validation contracts, inference, explainability, trajectory summaries, and optional AI narrative |
| `app/index.html` | Responsive researcher-facing audit studio |
| `model/train.py` | Deterministic split, NHANES processing, model training, evaluation, artifacts, and benchmark generation |
| `model/audit_and_reweight.py` | Dataset-agnostic post-stratification and weight-capping utility |
| `artifacts/model_a.joblib` | Saved raw baseline model |
| `artifacts/model_b.joblib` | Saved population-reweighted model |
| `artifacts/split_indices.npz` | Exact fit, calibration, and test row indices |
| `artifacts/shap_background.csv` | Saved SHAP background sample |
| `artifacts/training_summary.json` | Machine-readable source of truth for the generated run |
| `BENCHMARK.md` | Generated evaluation methodology and results |
| `data/DATA_DEBT_REPORT.md` | Distributional findings, limitations, and collection priorities |
| `data/datasheets/` | Source-specific dataset notes |
| `tests/` | API safety, artifact exposure, trajectory, AI-boundary, audit, and documentation tests |

---

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime readiness, artifact schema, model variant, and validation status |
| `GET` | `/v1/model-card` | Generated model card and reproducibility metadata |
| `GET` | `/v1/representation-audit` | Source-versus-reference strata and weight audit |
| `POST` | `/v1/assess` | Run one strict fictional profile through Models A and B with local explanation |
| `POST` | `/v1/counterfactual` | Perform a bounded one-feature sensitivity test |
| `GET` | `/v1/explain/global` | Return global Model B SHAP sensitivity |
| `GET` | `/v1/trajectory/schema` | Publish the longitudinal record contract |
| `POST` | `/v1/trajectory` | Validate and summarize an identifier-free daily record |
| `POST` | `/v1/trajectory/explain` | Optionally explain an aggregate-only trajectory summary |
| `GET` | `/v1/data-debt-report` | Expose the documented data-debt report |

Interactive OpenAPI documentation is available at `/docs` while the service is running.

---

## Run locally

### Requirements

- Windows, macOS, or Linux;
- Python 3.12 recommended; and
- the supplied model artifacts, or the raw files required to regenerate them.

### Windows PowerShell

```powershell
git clone https://github.com/shamaiem10/Trace.git
cd Trace

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

python -m pytest -q
python -m uvicorn api.main:app --reload
```

Open:

- application: <http://127.0.0.1:8000>
- API documentation: <http://127.0.0.1:8000/docs>
- health endpoint: <http://127.0.0.1:8000/health>

If port 8000 is occupied:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Do not open `app/index.html` directly; the interface must be served by FastAPI so its live API requests work.

### Regenerate the models and reports

```powershell
python -m model.train --data-dir data/raw --artifact-dir artifacts
```

This regenerates Model A, Model B, split indices, SHAP background, the training summary, and `BENCHMARK.md` together. Restart the API after regeneration.

### Optional aggregate-only AI explanation

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=your_private_key
TRACE_OPENAI_MODEL=
```

The API uses the configured override or defaults to `gpt-4.1-mini`. The `.env` file is ignored by Git. Never commit API keys.

---

## Reproducibility

The generated artifact records:

- random seed `42`;
- exact split sizes and row-index artifact;
- Python, NumPy, and scikit-learn versions;
- SHA-256 of the labelled input;
- model hyperparameters in source control;
- saved explanation background;
- reference respondent count;
- raw weight range, cap, and capped fraction;
- validation status; and
- all reported held-out metrics.

At API startup, Trace checks that the installed NumPy and scikit-learn versions match the versions recorded with the artifacts. This prevents silently loading version-incompatible serialized models.

---

## Verification

The current repository passes:

```text
20 tests passed
Python modules compile successfully
No broken Python package requirements
Browser model inspection works
Browser global SHAP interaction works
No browser console errors in the verified flows
```

The tests cover strict profile validation, identifier rejection, validation-status disclosure, generated model-card and audit artifacts, static application delivery, counterfactual domains, trajectory data quality, aggregate-only AI boundaries, OpenAI response parsing, SHAP availability behavior, weighting stability, BMI boundaries, and benchmark rendering.

---

## Responsible scope

### What Trace supports

- demonstrating a reproducible research-model audit workflow;
- comparing one supplied labelled feature distribution with one selected public reference distribution;
- studying sensitivity to post-stratification choices;
- inspecting the behavior of two saved research models;
- publishing reusable schemas, artifacts, and evaluation code; and
- making missing evidence visible.

### What Trace does not support

- diagnosing or screening a person for endometriosis;
- estimating an individual’s clinical risk or prognosis;
- recommending treatment, medication, or care;
- claiming that NHANES is the perfect target population for every use case;
- repairing unverified or biased labels through weighting;
- inferring causality from SHAP values;
- treating an internal test result as external clinical validation; or
- deploying the model in patient care.

### Required next steps toward clinical relevance

1. Establish source provenance, consent, licence, and clinically adjudicated outcome definitions.
2. Collect longitudinal person-level data with recruitment metadata and prespecified missing-data policy.
3. Define the actual target population and validate whether the reference variables are appropriate.
4. Evaluate discrimination, calibration, subgroup performance, and harms across multiple external cohorts.
5. Conduct prospective validation with women’s-health experts and affected communities.
6. Complete privacy, ethics, governance, security, and applicable regulatory review.

---

## Hack-Nation alignment

The challenge asks teams to build one reusable layer that future researchers can reproduce, extend, and combine into larger systems. Trace focuses on **Layer 2: AI Model Infrastructure**, with a supporting Layer 3 demonstration.

| Strong-submission criterion from the brief | Evidence in Trace |
| --- | --- |
| Publish reusable models and evaluation pipelines | Saved Models A/B, deterministic splits, training code, generated benchmark |
| Solve one defined infrastructure problem transparently | Population-mismatch audit and fit-only post-stratification workflow |
| Make assumptions and evaluation choices visible | Data-debt report, model card, datasheets, validation flags, metric definitions |
| Share reproducible code and documentation | Pinned packages, source hash, artifact metadata, tests, this handoff |
| Avoid unsupported medical claims | Research-only contracts and UI language throughout |
| Build a convincing application | Live audit studio driven by the backend pipeline |

---

## Acknowledgements

- Hack-Nation and its collaborating communities for the Challenge 05 brief.
- The U.S. National Center for Health Statistics for the NHANES public-use survey components.
- The open-source maintainers of FastAPI, pandas, NumPy, scikit-learn, SHAP, joblib, Uvicorn, pytest, and related dependencies.

Dataset source terms and applicable public-use requirements remain authoritative. The supplied labelled dataset’s provenance and licence are unconfirmed and must be resolved before redistribution or expanded use.

---

<div align="center">

### Evidence before confidence.

**Trace makes the data, assumptions, correction, uncertainty, explanation, and limits visible together.**

</div>
