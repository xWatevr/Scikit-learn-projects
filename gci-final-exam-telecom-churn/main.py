import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

client = pd.read_csv(r'E:\Project_GCI\final_assignment\telecom\Client.csv')
record = pd.read_csv(r'E:\Project_GCI\final_assignment\telecom\Record.csv')

df = client.merge(record, on="Customer_ID", how="left")

#print(client["Customer_ID"].nunique(), len(client))
#print(record["Customer_ID"].nunique(), len(record))

#print(client["Customer_ID"].duplicated().sum())
#print(record["Customer_ID"].duplicated().sum())

print("Client dataset")
print(pd.DataFrame(client).head())

print("Records dataset")
print(pd.DataFrame(record).head())

print('')
print(df.shape)
print(df.head())
print(df.info())

missing_counts = df.isna().sum()
top_missing = missing_counts.sort_values(ascending=False).head(10)
print(top_missing)

sns.barplot(x=top_missing.values, y=top_missing.index)
plt.xlabel('Number of missing values')
plt.title('Top 20 columns by number of missing values')
plt.show()

# Counts and proportions
normalized_churn = df['churn'].value_counts(normalize=True) * 100
print(normalized_churn)

# Bar chart
counts = df['churn'].value_counts().sort_index()
labels = ['stayed (0)', 'churned (1)']

plt.figure(figsize=(5, 4))

bars = plt.bar(labels, counts.values)

plt.title('Churn distribution')
plt.ylabel('Number of customers')

for bar, pct in zip(bars, normalized_churn.values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{pct:.1f}%',
        ha='center',
        va='bottom'
    )

plt.show()



# equipment age vs churn

# Split eqpdays into two groups based on churn
stayed = df[df['churn'] == 0]['eqpdays']
churned = df[df['churn'] == 1]['eqpdays']

# Compare the averages
print(f'Mean equipment age (stayed):  {stayed.mean():.1f} days')
print(f'Mean equipment age (churned): {churned.mean():.1f} days')

# Overlay the two distributions
plt.figure(figsize=(8, 5))
plt.hist(stayed, bins=30, alpha=0.5, label='stayed')
plt.hist(churned, bins=30, alpha=0.5, label='churned')
plt.xlabel('Equipment age (days)')
plt.ylabel('Number of customers')
plt.title('Equipment age: stayers vs churners')
plt.legend()
plt.show()


# Here we gather several not relevant features together
plot_df = df[[
    'months',
    'rev_Mean',
    'mou_Mean',
    'totmrc_Mean',
    'eqpdays',
    'churn'
]].dropna().copy()

# Log-transform highly skewed positive variables
plot_df['log_rev_Mean'] = np.log1p(plot_df['rev_Mean'])
plot_df['log_mou_Mean'] = np.log1p(plot_df['mou_Mean'])
plot_df['log_totmrc_Mean'] = np.log1p(plot_df['totmrc_Mean'])
plot_df['log_eqpdays'] = np.log1p(plot_df['eqpdays'])

plot_df['churn_label'] = plot_df['churn'].map({
    0: 'stayed',
    1: 'churned'
})

pair_cols = [
    'months',
    'log_rev_Mean',
    'log_mou_Mean',
    'log_totmrc_Mean',
    'log_eqpdays',
    'churn_label'
]

pair_sample = plot_df[pair_cols].sample(n=5000, random_state=42)

sns.pairplot(
    pair_sample,
    hue='churn_label',
    corner=True,
    diag_kind='kde',
)

plt.suptitle('Pairwise view of tenure, revenue, usage, equipment age, and churn', y=1.02)
plt.show()




work_data = df.copy()

# Fill important columns before feature engineering
work_data['ovrrev_Mean'] = work_data['ovrrev_Mean'].fillna(0)
work_data['rev_Mean'] = work_data['rev_Mean'].fillna(0)

# Data service failures
work_data['data_fail_Mean'] = (
    work_data['blck_dat_Mean'].fillna(0) +
    work_data['drop_dat_Mean'].fillna(0)
)

# Voice service failures
work_data['voice_fail_Mean'] = (
    work_data['blck_vce_Mean'].fillna(0) +
    work_data['drop_vce_Mean'].fillna(0)
)

# Total service failures
work_data['total_fail_Mean'] = (
    work_data['data_fail_Mean'] +
    work_data['voice_fail_Mean']
)

# Total service failure rate
work_data["total_fail_rate"] = np.where(
    work_data["attempt_Mean"].fillna(0) > 0,
    work_data["total_fail_Mean"] / work_data["attempt_Mean"].fillna(0),
    0
)

# Voice activity
work_data["voice_activity_Mean"] = (
    work_data["plcd_vce_Mean"].fillna(0) +
    work_data["recv_vce_Mean"].fillna(0)
)

# Call success rate
work_data["call_success_rate"] = np.where(
    work_data["attempt_Mean"].fillna(0) > 0,
    work_data["complete_Mean"].fillna(0) / work_data["attempt_Mean"].fillna(0),
    0
)

# Peak voice call ratio
work_data["peak_voice_ratio"] = (
    work_data["peak_vce_Mean"].fillna(0) /
    (
        work_data["peak_vce_Mean"].fillna(0) +
        work_data["opk_vce_Mean"].fillna(0) +
        1
    )
)

# Missed voice call rate
work_data['unanswered_voice_rate'] = np.where(
    work_data['plcd_vce_Mean'].fillna(0) > 0,
    work_data['unan_vce_Mean'].fillna(0) / work_data['plcd_vce_Mean'].fillna(0),
    0
)

# Received voice call rate
work_data['received_voice_rate'] = np.where(
    work_data['plcd_vce_Mean'].fillna(0) > 0,
    work_data['recv_vce_Mean'].fillna(0) / work_data['plcd_vce_Mean'].fillna(0),
    0
)

# Share of extra charges in monthly revenue
work_data['overage_revenue_ratio'] = np.where(
    work_data['rev_Mean'] > 0,
    work_data['ovrrev_Mean'] / work_data['rev_Mean'],
    0
)

work_data['mou_3_6_ratio'] = np.where(
    work_data['avg6mou'].fillna(0) > 0,
    work_data['avg3mou'].fillna(0) / work_data['avg6mou'].fillna(0),
    0
)

work_data['rev_3_6_ratio'] = np.where(
    work_data['avg6rev'].fillna(0) > 0,
    work_data['avg3rev'].fillna(0) / work_data['avg6rev'].fillna(0),
    0
)

work_data['qty_3_6_ratio'] = np.where(
    work_data['avg6qty'].fillna(0) > 0,
    work_data['avg3qty'].fillna(0) / work_data['avg6qty'].fillna(0),
    0
)

# Data failure rate
work_data['data_fail_rate'] = np.where(
    work_data['plcd_dat_Mean'].fillna(0) > 0,
    work_data['data_fail_Mean'] / work_data['plcd_dat_Mean'].fillna(0),
    0
)

# Data success rate
work_data['data_success_rate'] = np.where(
    work_data['plcd_dat_Mean'].fillna(0) > 0,
    work_data['comp_dat_Mean'].fillna(0) / work_data['plcd_dat_Mean'].fillna(0),
    0
)

# Average minutes per completed data connection
work_data['avg_minutes_per_completed_data'] = np.where(
    work_data['comp_dat_Mean'].fillna(0) > 0,
    work_data['mou_cdat_Mean'].fillna(0) / work_data['comp_dat_Mean'].fillna(0),
    0
)

# Peak data ratio
work_data['peak_data_ratio'] = np.where(
    (work_data['peak_dat_Mean'].fillna(0) + work_data['opk_dat_Mean'].fillna(0)) > 0,
    work_data['peak_dat_Mean'].fillna(0) /
    (work_data['peak_dat_Mean'].fillna(0) + work_data['opk_dat_Mean'].fillna(0)),
    0
)

# Unanswered data rate
work_data['unanswered_data_rate'] = np.where(
    work_data['plcd_dat_Mean'].fillna(0) > 0,
    work_data['unan_dat_Mean'].fillna(0) / work_data['plcd_dat_Mean'].fillna(0),
    0
)
# Overage + roaming pressure
work_data["extra_cost_pressure"] = (
    work_data["ovrrev_Mean"].fillna(0) +
    work_data["roam_Mean"].fillna(0)
)
# Customer care intensity
work_data["care_calls_per_month"] = np.where(
    work_data["months"].fillna(0) > 0,
    work_data["custcare_Mean"].fillna(0) / work_data["months"].fillna(0),
    0
)

# Total usage intensity
work_data["usage_intensity"] = (
    work_data["mou_Mean"].fillna(0) +
    work_data["datovr_Mean"].fillna(0)
)

cols_to_delete = [
    'recv_sms_Mean',
    'iwylis_vce_Mean',
    'Customer_ID',
    'forgntvl',
    'ethnic'
]

work_data = work_data.drop(columns=cols_to_delete)

# Correlation of all numeric features with churn
churn_corr = (
    work_data
    .corr(numeric_only=True)[['churn']]
    .sort_values(by='churn', ascending=False)
)

plt.figure(figsize=(6, 18))
sns.heatmap(
    churn_corr,
    annot=True,
    cmap='coolwarm',
    center=0
)

plt.title('Correlation with Churn')
plt.show()

print("Dateset info", work_data.info())

missing = work_data.isna().mean().sort_values(ascending=False) * 100

print("Data with missing values", missing[missing > 0])

missing_cols_to_drop = [
    "numbcars",
    "dwllsize",
    "HHstatin",
    "ownrent",
    "dwlltype",
    "infobase"
]
work_data = work_data.drop(columns=missing_cols_to_drop)

num_cols = work_data.select_dtypes(include=["int64", "float64"]).columns
str_cols = work_data.select_dtypes(include=["object", "str", "string"]).columns

for col in num_cols:
    work_data[col] = work_data[col].fillna(work_data[col].median())

for col in str_cols:
    work_data[col] = work_data[col].fillna("Unknown")


print("Number of missing values")
print(work_data.isna().sum().sum())

# One-hot encoding
work_data_encoded = pd.get_dummies(
    work_data,
    columns=str_cols,
)

# Check
print("Shape after encoding:", work_data_encoded.shape)



# Model assembling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier



X = work_data_encoded.drop(columns=["churn"])
y = work_data_encoded["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    ),

    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=42
        ),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=20,
            random_state=42
        ),
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        gamma=0.2,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=3.0,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": auc
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="ROC-AUC", ascending=False)

print("\nModel comparison:")
print(results_df.round(5))


best_model = models["XGBoost"]

importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(importances.head(30))

plt.figure(figsize=(15, 6))

sns.barplot(
    data=importances.head(20),
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Feature Importances - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()



from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=perm_df.head(30),
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Permutation Importances - XGBoost ROC-AUC")
plt.xlabel("Decrease in ROC-AUC when feature is shuffled")
plt.ylabel("Feature")
plt.show()


y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Stayed", "Predicted Churned"],
    yticklabels=["Actual Stayed", "Actual Churned"]
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

features = ["months", "eqpdays", "rev_Mean", "mou_Mean"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(
        data=work_data,
        x="churn",
        y=col,
        ax=axes[i],
        showfliers=False
    )

    axes[i].set_title(f"{col} vs Churn")
    axes[i].set_xlabel("Churn")
    axes[i].set_ylabel(col)
    axes[i].set_xticklabels(["Stayed (0)", "Churned (1)"])

plt.tight_layout()
plt.show()

features_titles = {
    "overage_revenue_ratio": "Overage revenue ratio",
    "mou_3_6_ratio": "MOU 3/6 ratio",
    "totmrc_Mean": "Monthly recurring charge",
    "extra_cost_pressure": "Extra cost pressure"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (feature, title) in zip(axes, features_titles.items()):
    stayed = work_data[work_data["churn"] == 0][feature].dropna()
    churned = work_data[work_data["churn"] == 1][feature].dropna()

    ax.hist(stayed, bins=30, alpha=0.5, label="stayed")
    ax.hist(churned, bins=30, alpha=0.5, label="churned")

    ax.set_title(f"{title}: stayers vs churners")
    ax.set_xlabel(title)
    ax.set_ylabel("Number of customers")
    ax.legend()

plt.tight_layout()
plt.show()

all_churn_proba = best_model.predict_proba(X)[:, 1]

customer_risk = pd.DataFrame({
    "Customer_ID": df.loc[X.index, "Customer_ID"].values,
    "churn_probability": all_churn_proba
})

# Convert probability into percentile
customer_risk["churn_risk_percentile"] = (
    customer_risk["churn_probability"].rank(pct=True) * 100
)

customer_risk = customer_risk.sort_values(
    by="churn_risk_percentile",
    ascending=False
)

customer_risk["risk_group"] = pd.cut(
    customer_risk["churn_risk_percentile"],
    bins=[0, 30, 60, 85, 100],
    labels=["Low risk", "Medium risk", "High risk", "Critical risk"],
    include_lowest=True
)

print(customer_risk["risk_group"].value_counts())

print(customer_risk.head(20))

# Add predicted churn probability to the dataset
work_data_for_business = work_data.copy()
work_data_for_business["churn_probability"] = best_model.predict_proba(X)[:, 1]

# Create churn risk groups
work_data_for_business["risk_group"] = pd.qcut(
    work_data_for_business["churn_probability"],
    q=[0, 0.30, 0.60, 0.85, 1.0],
    labels=["Low risk", "Medium risk", "High risk", "Critical risk"]
)


average_income_for_person = work_data['rev_Mean'].mean()
print(f"Average income per person: ¥{average_income_for_person}")