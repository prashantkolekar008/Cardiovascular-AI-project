from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH =  Path(
    r"C:\Users\Lenovo\Desktop\dataset\extention_of_z_alizadeh_sani_dataset.xlsx"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents = True ,exist_ok=True) 

# Load dataset
df = pd.read_excel(DATASET_PATH)

# Exclude every target column from model inputs
TARGET_COLUMNS = ["Cath", "LAD", "LCX", "RCA"]
X = df.drop(columns=TARGET_COLUMNS)

# Identify numerical and categorical features
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)

# Train one model for each coronary vessel
for target in ["LAD", "LCX", "RCA"]:
    print(f"\n{'=' * 45}")
    print(f"Training {target} stenosis model")
    print("=" * 45)

    # Remove rows with missing target labels, if any
    valid = df[target].notna()
    X_target = X.loc[valid]
    y_raw = df.loc[valid, target].astype(str).str.strip()

    # Convert known labels to binary values
    label_map = {
        "Normal": 0,
        "Stenotic": 1,
    }

    y = y_raw.map(label_map)

    # Stop if unexpected labels exist
    if y.isna().any():
        unexpected = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(
            f"Unexpected labels in {target}: {unexpected}. "
            "Inspect the dataset before training."
        )

    y = y.astype(int)

    # Split data while preserving class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Build complete model pipeline
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

    # Train
    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Records used: {len(y)}")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(
        f"Recall   : "
        f"{recall_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.3f}")

    # Save trained model
    model_path = MODEL_DIR / f"{target.lower()}_model.joblib"
    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

print("\nAll three vessel models have been trained.")
print("Educational prototype only—not for medical diagnosis.")                  