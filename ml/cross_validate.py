from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project folders
ROOT = Path(__file__).resolve().parent.parent
DATASET_FOLDER = ROOT / "dataset"

excel_files = list(DATASET_FOLDER.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError(
        "Dataset nahi mili. Check karo ki Excel file dataset folder mein hai."
    )

df = pd.read_excel(excel_files[0])

# Target: CAD / Normal
y = df["Cath"].astype(str).str.strip().str.upper().map({"CAD": 1, "NORMAL": 0})

if y.isna().any():
    raise ValueError("Cath column mein unexpected labels hain.")

# Target leakage se bachne ke liye ye columns remove karo
X = df.drop(columns=["Cath", "LAD", "LCX", "RCA"])

# Numeric aur categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

# Numeric data processing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical data processing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
)

print("\nCardioVision-AI: 5-Fold Cross-Validation")
print("-" * 45)

for metric in scoring:
    scores = results[f"test_{metric}"]
    print(
        f"{metric.upper():12}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )

print("\nEducational prototype only.")
print("Not intended for medical diagnosis.")
