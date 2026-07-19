# Data Debt Report — structured_endometriosis_data.csv

## What this audit can support

This audit compares selected *feature distributions* in the labelled dataset with a survey-weighted NHANES reference population. It cannot validate diagnosis labels, establish causality, or make the model clinically diagnostic.

## Reference population

`RHQ_L.xpt`, `DEMO_L.xpt`, and `BMX_L.xpt` are joined on `SEQN`, then filtered to female respondents aged 18–49 with valid menstrual-regularity response and measured BMI. The current supplied files yield 1,206 complete respondents. Rates use `WTINT2YR`; unweighted rates are not used as population estimates.

## Identified data debt

- The labelled dataset has a much larger Normal-BMI share and much smaller Obese-BMI share than the NHANES reference distribution.
- The labelled dataset reports substantially more menstrual irregularity than the survey-weighted reference distribution.
- The labelled dataset's provenance and label-generation process are unconfirmed. Distributional correction cannot repair label quality.

## Mitigation implemented

The model uses post-stratification across BMI category × menstrual irregularity. Raw weights range from 0.0289 to 26.9259; to prevent a few observations dominating training, the implementation caps at five times the positive median (0.8412 before mean-one normalization). 29.0% of rows are capped.

## What future data collection should fix

Collect clinically verified labels and recruitment metadata, recruit across the BMI distribution (especially currently sparse high-BMI cells), validate menstrual-history wording, and compare results against more than one appropriate reference population. Do not deploy the model for clinical decisions without prospective validation and appropriate governance.
