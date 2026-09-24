# Telecom Customer Churn Prediction

**GCI Final Project | Binary Classification | Customer Risk Analysis**

This project was developed as my final assignment for the GCI course. It explores telecom customer data, compares machine learning classifiers, and groups customers by their predicted churn risk.

## Objective

Predict whether a customer will leave the telecom provider and explore patterns that could inform customer retention strategies.

The target variable is `churn`:

- **0** — the customer stayed.
- **1** — the customer churned.

## Data

The project uses two datasets, joined by `Customer_ID` with a left join:

| File | Role |
| --- | --- |
| `input/Client.csv` | Customer information |
| `input/Record.csv` | Customer service and usage records |

The analysis includes customer tenure, revenue, voice and data usage, service failures, equipment age, and other customer attributes. Both datasets are required to run the script.

## Project Workflow

### 1. Exploratory Data Analysis

- Inspect dataset structure and missing values.
- Examine the distribution of churn.
- Compare equipment age, tenure, revenue, and usage between customers who stayed and those who churned.
- Visualize feature distributions, pairwise relationships, and correlations with churn.

### 2. Data Preparation and Feature Engineering

- Remove selected identifiers and unused columns, as well as several columns with substantial missing data.
- Fill selected usage and revenue fields with zero, remaining numeric missing values with medians, and categorical missing values with `Unknown`.
- Apply one-hot encoding to categorical features.
- Create features describing service failure rates, call success, peak usage, extra charges, changes in usage and revenue, and customer care activity.

Examples include `total_fail_rate`, `call_success_rate`, `overage_revenue_ratio`, `mou_3_6_ratio`, and `extra_cost_pressure`.

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

The data is split into **80% training** and **20% test** sets, using stratification by churn and `random_state=42`.

Evaluation metrics:

- **ROC AUC** — calculated from predicted churn probabilities.
- **Accuracy** — overall proportion of correct predictions.
- **Precision** — proportion of predicted churners who actually churned.
- **Recall** — proportion of actual churners identified by the model.

The script prints a comparison table sorted by ROC AUC. XGBoost is explicitly selected in the code for the subsequent interpretation and risk analysis; this selection is not automatically based on the comparison table.

### 4. Model Interpretation

For XGBoost, the project examines:

- Built-in feature importance.
- Permutation importance using ROC AUC on the test set.
- A confusion matrix.

### 5. Customer Risk Groups

Customers are ranked by predicted churn probability and divided into four relative risk groups:

| Group | Approximate percentile range |
| --- | --- |
| Low risk | Bottom 30% |
| Medium risk | 30–60% |
| High risk | 60–85% |
| Critical risk | Top 15% |

These groups describe relative model scores within this dataset. They are not fixed probability thresholds or a clustering algorithm. They provide a starting point for exploring which customers might deserve further retention analysis.

## Project Files

| Path | Contents |
| --- | --- |
| `main.py` | Data analysis, preprocessing, training, evaluation, and risk grouping |
| `input/Client.csv` | Customer data |
| `input/Record.csv` | Usage and service records |
| `README.md` | Project documentation |
| `reports/` | Optional presentation or final report |

## How to Run

1. Place `Client.csv` and `Record.csv` in the `input` folder beside `main.py`.
2. Install the required libraries in your Python environment:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

3. If the script still contains the original absolute Windows paths, replace the two CSV-loading statements with:

```python
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent / "input"

client = pd.read_csv(INPUT_DIR / "Client.csv")
record = pd.read_csv(INPUT_DIR / "Record.csv")
```

4. Remove the unused import of `primitive_column_to_ndarray` from `pandas.core.interchange.from_dataframe`, if present. This internal pandas helper is not needed by the project.
5. Open a terminal in the project folder and run:

```bash
python main.py
```

The script prints data summaries, model metrics, feature importance, and customer risk information. Plots open during execution; close each plot window to continue. The current script displays results rather than exporting trained models or reports.

The pairplot samples 5,000 complete rows. If using a smaller dataset, change `sample(n=5000, ...)` to use `n=min(5000, len(plot_df))`.

## Results

The code generates a model comparison table with accuracy, precision, recall, and ROC AUC, together with XGBoost interpretation plots and customer risk summaries. Numerical scores are not listed here because execution results were not provided with this version of the script.

## Limitations and Next Steps

- Median imputation and one-hot encoding are performed before the train/test split. A stricter evaluation should split the data first and fit preprocessing only on the training set, using a pipeline.
- Comparing and selecting models repeatedly on the same test set can bias the final assessment. Cross-validation on the training set and a separate final holdout would improve evaluation.
- Risk groups are calculated for the entire dataset, including training customers. They are exploratory; use out-of-fold predictions or a separate unseen dataset for a more reliable assessment.
- Feature importance and correlations show predictive associations, not causal explanations of churn.
- Future work could evaluate probability calibration and test retention strategies against their costs and measured business outcomes.

## Skills Practiced

Tabular data analysis, visualization, missing-value handling, feature engineering, one-hot encoding, classification, model evaluation, feature interpretation, and translating model scores into an exploratory business analysis.
