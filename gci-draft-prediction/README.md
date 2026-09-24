# GCI Competition — Player Draft Prediction

**Binary classification · Exploratory data analysis · Feature engineering · ROC AUC**

A machine learning project developed during a competition in the GCI course I attended. The task is to estimate the probability that a player was drafted, using physical measurements, athletic test results, and player information.

## Course context

This repository documents my competition work from the GCI course. It includes an initial Random Forest solution and an extended version using additional feature engineering and CatBoost.

Through this project, I practiced inspecting tabular data, visualizing distributions and correlations, handling missing values, encoding categories, engineering features, evaluating classifiers with stratified cross-validation, and generating predictions for submission.

## Task and data

The target is `Drafted`: `1` indicates a drafted player and `0` indicates a player who was not drafted. Predictions are probabilities for class `1`.

| Dataset | Rows | Columns | Target available |
| --- | ---: | ---: | --- |
| `input/train.csv` | 2,781 | 17 | Yes |
| `input/test.csv` | 696 | 16 | No |

The training set contains 1,803 positive and 978 negative examples. Features include year, age, school, height, weight, 40-yard sprint, vertical and broad jumps, bench press repetitions, agility tests, and player position. `Id` is excluded from model inputs.

The supplied CSV files are included in this project package. Their original source and redistribution terms are not documented here; check the competition terms before publishing the datasets.

## Solutions

| Script | Active model | Main characteristics |
| --- | --- | --- |
| `main.py` | Random Forest | EDA plots, mean imputation, label encoding, BMI; 100 trees, maximum depth 5 |
| `enhanced_model.py` | CatBoost | Adds school frequency encoding; 500 iterations, learning rate 0.03, depth 5; averages test probabilities across five folds |

Both scripts use five-fold `StratifiedKFold` with shuffling and `random_state=42`. Evaluation uses ROC AUC calculated from predicted probabilities. BMI is calculated as weight divided by squared height.

The enhanced script also contains inactive experimental configurations for Gradient Boosting, XGBoost, and LightGBM. CatBoost, XGBoost, and LightGBM are separate libraries, even though they can be used alongside scikit-learn tools.

## Files

- `main.py`: initial model and exploratory plots.
- `enhanced_model.py`: extended feature set and active CatBoost model.
- `input/train.csv`, `input/test.csv`: supplied competition data.
- `requirements.txt`: dependencies for the two active scripts.
- `.gitignore`: excludes local environments, caches, and generated predictions.
- `outputs/`: created automatically when a script runs.

## Run locally

Use Python 3.10 or newer. Open a terminal in this project's directory:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run either solution:

```bash
python -m pip install -r requirements.txt
python main.py
python enhanced_model.py
```

The initial script opens plots; close each plot window to continue. Both scripts print per-fold and mean validation AUC. Generated files are `outputs/submission_random_forest.csv` and `outputs/submission_catboost.csv`.

An official `sample_submission.csv` was not supplied. These portable scripts construct an `Id,Drafted` submission using test IDs in their original order. Verify this format against the official competition template before submitting.

Dependency versions are not pinned; this is a learning project rather than a fully locked reproduction environment. To activate the XGBoost or LightGBM experiments, install the corresponding package first.

## Results and interpretation

The original experimental comments record these values:

| Configuration | AUC noted in source comments |
| --- | ---: |
| Gradient Boosting | 0.8265 |
| XGBoost | 0.8310 |
| CatBoost | 0.8319 |
| LightGBM | 0.8279 |

These are historical notes from the source code, not independently verified benchmark results. The supplied files do not establish whether each value is a validation score or a competition leaderboard score. No final ranking is claimed.

## Current limitations

- Mean imputation and, in the enhanced script, school frequency encoding are fitted before cross-validation. Validation folds therefore influence preprocessing statistics. For stricter evaluation, fit these transformations on each training fold only.
- `LabelEncoder` imposes numeric ordering on nominal categories and cannot transform unseen test categories. A future version could use an encoder with unknown-category handling or CatBoost's native categorical features.
- The initial Random Forest script exports predictions from the last fold only. The enhanced script averages predictions from all five folds.
- Some exploratory expressions were written in notebook style and do not print their results when run as a Python script.

## Packaging changes

The competition modeling logic has been retained. Packaging removes an unused invalid `statistics.LinearRegression` import, replaces machine-specific Windows paths with paths relative to each script, constructs the missing submission template from test IDs, and writes predictions to `outputs/`.
