# Mammographic Mass Classification

Predicting the probability that a mammographic mass is malignant, as a decision-support aid for biopsy decisions.

*Midterm project for ML Zoomcamp (DataTalks.Club).*

## Problem Description

When a mass is found on a mammogram, the radiologist must decide whether
to recommend a biopsy. Many masses referred for biopsy turn out to be
benign, so those patients undergo an invasive procedure that was not
necessary. The goal of this project is to estimate the probability that a
mass is malignant from its imaging characteristics and patient age, as a
second opinion that could help reduce unnecessary biopsies.

**Task:** binary classification (benign = 0, malignant = 1).

**Intended use:** a decision-support tool. The model outputs a malignancy
probability that a radiologist can consider alongside their own judgment.
It does not replace clinical decision-making.

## Dataset

[UCI Mammographic Mass dataset](https://archive.ics.uci.edu/dataset/161/mammographic+mass)
(CC BY 4.0), 961 cases.

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Shape | Mass shape |
| Margin | Mass margin |
| Density | Mass density |
| Severity | **Target**: benign / malignant |

Some values are missing and are handled during cleaning (see the notebook).

### Why BI-RADS is excluded

The dataset also contains a `BI-RADS` assessment. It is the radiologist's
own overall rating of the case, and the UCI documentation marks it as
non-predictive. Using it as a feature would leak the radiologist's
judgment into the model (data leakage), so it is excluded.

## Limitations

- The data comes from a single institution (2003–2006), so the model is
  **not ready for clinical use** and has not been validated on other
  populations or modern equipment.
- The dataset is small, so cross-validation is used when comparing models.