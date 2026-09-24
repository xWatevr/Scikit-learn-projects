from pathlib import Path
import numpy as np  # Library for numerical computing and array operations
import pandas as pd  # Library for handling tabular data
import matplotlib.pyplot as plt  # Library for basic data visualization
import seaborn as sns  # Library for statistical data visualization
from sklearn.preprocessing import LabelEncoder  # Encoder for converting categorical variables to numeric labels
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.model_selection import StratifiedKFold  # Class for stratified K-fold cross-validation
from sklearn.metrics import roc_auc_score  # Metric function for computing ROC AUC

pd.set_option('display.max_columns', None) # что бы не обрезались колонки
pd.set_option('display.width', None)  # Ограничивает, сколько текста помещается в строке
pd.set_option('display.max_colwidth', None) # Максимальная длина текста в ОДНОЙ ячейке

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
train = pd.read_csv(INPUT_DIR / "train.csv")
test = pd.read_csv(INPUT_DIR / "test.csv")
# Construct the expected Id/Drafted format; verify against the official template.
sample_sub = pd.DataFrame({"Id": test["Id"].copy(), "Drafted": 0.0})


# First, let's check the shape of the dataset   q
'Train:', train.shape
'Test:', test.shape


# Next, let's take a look at the first five rows of the training data
"Retrieve a table:", sample_sub.head()

#You can view detailed information about a pandas DataFrame by using .info()
train.info()

train.describe()
test.describe()
train.duplicated().sum()

# First, let's check for missing values
train.isnull().sum()
test.isnull().sum()


# Next, let's look at how many players were drafted.
drafted_counts = train['Drafted'].value_counts()
'''
plt.figure(figsize=(8, 6))
plt.bar(drafted_counts.index.astype(str), drafted_counts.values)
plt.title('Distribution of Drafted', fontsize=16)
plt.xlabel('Drafted', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
'''
# We can see that there are more players with Drafted = 1 than those with Drafted = 0. Let's check the propotions

drafted_percentage = train['Drafted'].value_counts(normalize=True) * 100
f"Percentage of 0: {drafted_percentage.get(0, 0):.2f}%"
f"Percentage of 1: {drafted_percentage.get(1, 0):.2f}%" # возьми процент для класса 1, а если класса 1 нет — верни 0

# Next, we will continue EDA for the other features.
# First, let's visualize the numerical variables excluding the "Id" and "Drafted" columns.

# Extraxt numeric columns (excluding the 'Id' and 'Drafted' columns)
numeric_cols = train.select_dtypes(include=['number']).columns
numeric_cols = numeric_cols.drop(['Id', 'Drafted'])

# Plot
num_cols = len(numeric_cols)
cols = 3
rows = (num_cols + cols - 1) // cols
'''
plt.figure(figsize=(5 * cols, 4 * rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, cols, i)
    plt.hist(train[col].dropna(), bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
'''

# Let's examine the correlation matrix of the numerical features using a heatmap
# Extract numeric columns (excluding the 'Id' and ''Drafted' columns)
numeric_cols = train.select_dtypes(include=['number']).drop(['Id', 'Drafted'], axis=1)

# Compute the correlation matrix
corr_matrix = numeric_cols.corr()

# Plot a heatmap
'''
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True, #Показывать числа внутри клеток.
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5
)

plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.show()

# Plot a boxplot
plt.figure(figsize=(6, 3))
sns.boxplot(x=train['Sprint_40yd'])

plt.title('Box Plot of Sprint_40yd', fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
'''

# Next, let's visualize the categorical features.
# Extract categorical features
categorical_cols = train.select_dtypes(include=['object', 'category', 'string']).columns

# Get the number of unique categories in each column
levels_count = {col: train[col].nunique() for col in categorical_cols}
for col, count in levels_count.items():
    print(f"{col}: {count} levels")
#'School' has 236 unique values, which is too many to visualize clearly, so we will exclude it from the plots here

# Extract categorical features (object or category) and exclude the 'School' columns
categorical_cols = train.select_dtypes(include=['object', 'category', 'string']).columns
categorical_cols = categorical_cols.drop('School')

# Prepare for plotting
num_cols = len(categorical_cols)
rows = 1
cols = num_cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5))

if cols == 1:
    axes = [axes]
else:
    axes = axes.flatten() # Если графиков много: превращает массив графиков в удобный одномерный список
'''
# Count plots for each categorical features
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=train, order=train[col].value_counts().index, ax=axes[i])
    #order=train[col].value_counts().index Сортирует категории:от самых частых, к самым редким.
    axes[i].set_title(f'Count Plot of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
'''
# There are missing values in Age, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump,
# Agility_3cone, and Shuttle. In this notebook, we will impute these missing values using the mean of each column
# Drop unused columns


# NEW
school_freq = train["School"].value_counts(normalize=True)
train["School_freq"] = train["School"].map(school_freq)
test["School_freq"] = test["School"].map(school_freq).fillna(0)
train = train.drop(columns=["Id", "School"])
test = test.drop(columns=["Id", "School"])


# missing columns will be replaced by similar values
cols_to_fill = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                'Broad_Jump', 'Agility_3cone', 'Shuttle']
for col in cols_to_fill:
    mean_value = train[col].mean()
    train[col] = train[col].fillna(mean_value)
    test[col] = test[col].fillna(mean_value)
#-----------------------------------------------------------------

# Let's check whether there are any missing values.
train.isnull().sum()
test.isnull().sum()

numeric_cols = train.select_dtypes(include=['number']).columns
numeric_cols = numeric_cols.drop(['Drafted'])
'''
# NEW
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Drafted', y=col, data=train)
    plt.title(f'{col} by Drafted')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
#-----------------------------------------------------------
'''

# Next, let's convert the categorical features into numerical values so they can be used as model inputs.
# Most machine learning models can only accept numerical data.
#
# Here, we will use a technique called label encoding.
# Label-encode the categorical features
label_encoders = {}
for c in ["Player_Type", "Position_Type", "Position"]:
    label_encoders[c] = LabelEncoder()
    train[c] = label_encoders[c].fit_transform(train[c].astype(str))
    test[c] = label_encoders[c].transform(test[c].astype(str))

# NEW
for df in [train, test]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

selected_features = [
    "Year",
    "Age",
    "Height",
    "Weight",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
    "School_freq",
    "BMI",

    "Player_Type",
    "Position_Type",
    "Position"
]

# Split into features and target
X = train[selected_features]
y = train["Drafted"]

test_X = test[selected_features]

# Set up the model
'''
from sklearn.ensemble import GradientBoostingClassifier # 0.8265
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=0
)

from xgboost import XGBClassifier # 0.831
model = XGBClassifier(
    n_estimators=450,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
'''
from catboost import CatBoostClassifier # 0.8319
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=5,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0 
)

'''
from lightgbm import LGBMClassifier # 0.8279
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=2,
    num_leaves=20,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=42
)
'''
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store scores
auc_scores = []

# Train and evaluate using Startified K-fold cross-validation
test_pred_proba_mean = np.zeros(len(test))
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Validation prediction & scoring
    y_valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_pred_proba)
    auc_scores.append(auc)

    test_pred_proba_mean += model.predict_proba(test_X)[:, 1] / skf.n_splits
    print(f"  AUC: {round(auc, 4)}")

# Print the mean AUC
mean_auc = np.mean(auc_scores)
print("\nAverage Validation AUC:", round(mean_auc, 4))

submission = sample_sub.copy()
submission["Drafted"] = test_pred_proba_mean
submission.to_csv(OUTPUT_DIR / "submission_catboost.csv", index=False)
