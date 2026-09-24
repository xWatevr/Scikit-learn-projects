# GCI Competition — Player Draft Prediction

**Binary classification · Exploratory data analysis · Feature engineering · ROC AUC**

A machine learning project developed during a competition in the GCI course I attended. The task is to estimate the probability that a player was drafted, using physical measurements, athletic test results, and player information.


## Course context

This repository documents my competition work from the GCI course. It includes an initial Random Forest solution and an extended version using additional feature engineering and CatBoost.

Through this project, I practiced inspecting tabular data, visualizing distributions and correlations, handling missing values, encoding categories, engineering features, evaluating classifiers with stratified cross-validation, and generating predictions for submission.

## Task and data

The target is `Drafted`: `1` indicates a drafted player and `0` indicates a player who was not drafted.

| Dataset | Rows | Columns | Target available |
| --- | ---: | ---: | --- |
| `input/train.csv` | 2,781 | 17 | Yes |
| `input/test.csv` | 696 | 16 | No |


## Solutions

| Script | Active model | Main characteristics |
| --- | --- | --- |
| `main.py` | Random Forest | EDA plots, mean imputation, label encoding, BMI; 100 trees, maximum depth 5 |
| `enhanced_model.py` | CatBoost | Adds school frequency encoding; 500 iterations, learning rate 0.03, depth 5; averages test probabilities across five folds |

Both scripts use five-fold `StratifiedKFold` with shuffling and `random_state=42`. Evaluation uses ROC AUC calculated from predicted probabilities. BMI is calculated as weight divided by squared height.

The enhanced script also contains inactive experimental configurations for Gradient Boosting, XGBoost, and LightGBM. CatBoost, XGBoost, and LightGBM are separate libraries, even though they can be used alongside scikit-learn tools.


## Results and interpretation

The original experimental comments record these values:

| Configuration | AUC noted in source comments |
| --- | ---: |
| Gradient Boosting | 0.8265 |
| XGBoost | 0.8310 |
| CatBoost | 0.8319 |
| LightGBM | 0.8279 |
