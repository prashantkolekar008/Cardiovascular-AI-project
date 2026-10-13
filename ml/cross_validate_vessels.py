# This code LAD, LCX aur RCA models ki 5-fold cross-validation karega.

from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project and dataset paths
BASE_DIR = Path(__file__).resolve().parent.parent
dataset_folder = BASE_DIR / "dataset"
excel_files = list(dataset_folder.glob("*.xlsx")) if dataset_folder.exists() else []
DATASET_PATH = excel_files[0] if excel_files else (dataset_folder / "extention_of_z_alizadeh_sani_dataset.xlsx")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_PATH}. Please place the dataset in the 'dataset/' folder."
    )

df = pd.read_excel(DATASET_PATH)

targets = ["LAD", "LCX", "RCA"]
X = df.drop(columns=["Cath", "LAD", "LCX", "RCA"])

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_features,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

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

for target in targets:
    print("\n" + "=" * 40)
    print(f"5-Fold Cross-Validation: {target}")
    print("=" * 40)

    valid = df[target].notna()
    X_target = X.loc[valid]
    labels = df.loc[valid, target].astype(str).str.strip()

    y = labels.str.lower().map({
        "normal": 0,
        "stenotic": 1,
    })

    if y.isna().any():
        raise ValueError(
            f"Unexpected target labels in {target}: "
            f"{labels[y.isna()].unique().tolist()}"
        )

    y = y.astype(int)

    results = cross_validate(
        clone(model),
        X_target,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    for metric in scoring:
        values = results[f"test_{metric}"]
        print(
            f"{metric.upper():10s}: "
            f"{values.mean():.3f} +/- {values.std():.3f}"
        )

print("\nCross-validation complete.")
print("Educational prototype only—not for medical diagnosis.")   