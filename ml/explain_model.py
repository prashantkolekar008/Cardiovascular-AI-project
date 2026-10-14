import pandas as pd
import joblib
import shap
from pathlib import Path
import matplotlib.pyplot as plt
# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "extention of Z-Alizadeh sani dataset.xlsx.xlsx"
MODEL_PATH = BASE_DIR / "models" / "cad_model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_excel(DATASET_PATH)

# Separate input features and target
X = df.drop(columns=["Cath", "LAD", "LCX", "RCA"])

# Preprocess the data using the saved model pipeline
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Select a small sample for explanation
sample = X_transformed[:5]

print("\nCardioVision AI - SHAP Explainable AI")
print("-" * 45)

# Create SHAP explainer
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    cad_shap_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    cad_shap_values = shap_values[:, :, 1]
else:
    cad_shap_values = shap_values

# Display feature contributions for the first sample
contributions = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Contribution": cad_shap_values[0]
})

contributions["Absolute Contribution"] = (
    contributions["SHAP Contribution"].abs()
)

contributions = contributions.sort_values(
    by="Absolute Contribution",
    ascending=False
)

print("\nTop 10 features influencing the first sample:\n")
print(contributions.head(10).to_string(index=False))

print("\nEducational prototype only.")
print("Not intended for medical diagnosis.")

# Plot top 10 feature contributions
top_features = contributions.head(10).sort_values(
    by="SHAP Contribution"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["SHAP Contribution"]
)

plt.axvline(x=0, linestyle="--")
plt.xlabel("SHAP Contribution")
plt.ylabel("Features")
plt.title("CardioVision AI - SHAP Feature Contributions")
plt.tight_layout()

# Save graph in project folder
output_path = BASE_DIR / "models" / "shap_feature_contributions.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nSHAP graph saved at: {output_path}")# Plot top 10 feature contributions
top_features = contributions.head(10).sort_values(
    by="SHAP Contribution"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["SHAP Contribution"]
)

plt.axvline(x=0, linestyle="--")
plt.xlabel("SHAP Contribution")
plt.ylabel("Features")
plt.title("CardioVision AI - SHAP Feature Contributions")
plt.tight_layout()

output_path = BASE_DIR / "models" / "shap_feature_contributions.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nSHAP graph saved at: {output_path}")