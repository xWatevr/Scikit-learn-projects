# Telecom Customer Churn Prediction

**GCI Final Project | Customer Risk Analysis**

This project was developed as my final assignment for the GCI course. It explores telecom customer data, compares machine learning classifiers, and groups customers by their predicted churn risk.

## Objective

Predict whether a customer will leave the telecom provider and explore patterns that could inform customer retention strategies.

The target variable is `churn`:

- **0** — the customer stayed
- **1** — the customer churned


## Project Workflow

### 1. Exploratory Data Analysis

- Inspect dataset structure and missing values
- Examine the distribution of churn
- Compare equipment age, tenure, revenue, and usage between customers who stayed and those who churned
- Visualize feature distributions, pairwise relationships, and correlations with churn

### 2. Data Preparation and Feature Engineering

- Remove selected identifiers and unused columns, as well as several columns with substantial missing data
- Fill selected usage and revenue fields with zero, remaining numeric missing values with medians, and categorical missing values with `Unknown`
- Apply one-hot encoding to categorical features
- Create features describing service failure rates, call success, peak usage, extra charges, changes in usage and revenue, and customer care activity

### 3. Model Comparison

Eight classifiers are trained and evaluated:

| Model | Library |
| --- | --- |
| Logistic Regression with StandardScaler | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Extra Trees | scikit-learn |
| Gradient Boosting | scikit-learn |
| Bagging | scikit-learn |
| AdaBoost | scikit-learn |
| XGBoost | XGBoost |

Evaluation metrics:

- **ROC AUC** — calculated from predicted churn probabilities
- **Accuracy** — overall proportion of correct predictions
- **Precision** — proportion of predicted churners who actually churned
- **Recall** — proportion of actual churners identified by the model

The script prints a comparison table sorted by ROC AUC. XGBoost is explicitly selected in the code for the subsequent interpretation and risk analysis; this selection is not automatically based on the comparison table.


### 4. Customer Risk Groups

Customers are ranked by predicted churn probability and divided into four relative risk groups:

| Group | Approximate percentile range |
| --- | --- |
| Low risk | Bottom 30% |
| Medium risk | 30–60% |
| High risk | 60–85% |
| Critical risk | Top 15% |

These groups describe relative model scores within this dataset. They are not fixed probability thresholds or a clustering algorithm. They provide a starting point for exploring which customers might deserve further retention analysis.


## Results

The code generates a model comparison table with accuracy, precision, recall, and ROC AUC, together with XGBoost interpretation plots and customer risk summaries. Numerical scores are not listed here because execution results were not provided with this version of the script.

## Limitations and Next Steps

- Median imputation and one-hot encoding are performed before the train/test split. A stricter evaluation should split the data first and fit preprocessing only on the training set, using a pipeline
- Comparing and selecting models repeatedly on the same test set can bias the final assessment. Cross-validation on the training set and a separate final holdout would improve evaluation
- Risk groups are calculated for the entire dataset, including training customers. They are exploratory; use out-of-fold predictions or a separate unseen dataset for a more reliable assessment
- Feature importance and correlations show predictive associations, not causal explanations of churn
- Future work could evaluate probability calibration and test retention strategies against their costs and measured business outcomes

## Skills Practiced

Tabular data analysis, visualization, missing-value handling, feature engineering, one-hot encoding, classification, model evaluation, feature interpretation, and translating model scores into an exploratory business analysis.
