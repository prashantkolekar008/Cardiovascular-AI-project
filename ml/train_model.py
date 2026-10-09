import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib

# 1. Dataset load karo
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
excel_files = list(dataset_folder.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError("Excel dataset dataset folder mein nahi mili.")

df = pd.read_excel(excel_files[0])

# 2. Target aur inputs select karo
target = "Cath"

# Target leakage se bachne ke liye ye columns input nahi banenge
leakage_columns = ["Cath", "LAD", "LCX", "RCA"]

X = df.drop(columns=leakage_columns)
y = (df[target].astype(str).str.strip().str.upper() == "CAD").astype(int)

print("Dataset shape:", df.shape)
print("Target counts:")
print(y.value_counts().rename(index={0: "Normal", 1: "CAD"}))

# 3. Categorical aur numeric columns identify karo
categorical_columns = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 6. Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluation
print("\n=== TEST RESULTS ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

print("\nClassification report:")
print(classification_report(
    y_test, y_pred, target_names=["Normal", "CAD"], zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save trained model
model_folder = project_folder / "models"
model_folder.mkdir(exist_ok=True)

joblib.dump(model, model_folder / "cad_model.joblib")

print("\nModel saved at models/cad_model.joblib")
print("Educational prototype only — not for medical diagnosis.")