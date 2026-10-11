
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(
    r"C:\Users\Lenovo\Desktop\dataset\extention_of_z_alizadeh_sani_dataset.xlsx"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATASET_PATH)

targets = ["LAD", "LCX", "RCA"]
X = df.drop(columns=["Cath", "LAD", "LCX", "RCA"])

for target in targets:
    print(f"\nExplaining {target} model...")

    model_path = MODEL_DIR / f"{target.lower()}_model.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue

    model = joblib.load(model_path)

    # Transform features using the fitted preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)

    # Use a small sample to keep SHAP computation manageable
    X_sample = X_transformed[:50]

    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Explain the positive class (Stenotic = 1)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)

    # SHAP versions may return different output shapes
    if isinstance(shap_values, list):
        values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    # Global feature importance
    plt.figure()
    shap.summary_plot(
        values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=10,
        show=False,
    )
    plt.title(f"{target} - Top 10 Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{target.lower()}_shap_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    print(f"Saved graph for {target}")

print(f"\nGraphs saved in: {OUTPUT_DIR}")
print("Educational prototype only—not for medical diagnosis.")     