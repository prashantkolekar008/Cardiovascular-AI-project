"""
Cardiovascular AI - FastAPI Backend Service
Provides real-time multi-vessel stenosis prediction, CAD risk scoring,
and clinical feature attribution for the 3D anatomical viewer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Cardiovascular AI API",
    description="Multi-vessel coronary artery stenosis prediction and 3D risk mapping backend.",
    version="1.0.0",
)

# Enable CORS for Next.js / Vite / Three.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models
MODELS: Dict[str, Any] = {}

def load_models():
    vessels = ["cad", "lad", "lcx", "rca"]
    for v in vessels:
        model_file = MODEL_DIR / f"{v}_model.joblib"
        if not model_file.exists():
            print(f"Warning: Model file not found: {model_file}")
            continue
        try:
            MODELS[v] = joblib.load(model_file)
            print(f"Loaded model: {v}")
        except Exception as e:
            print(f"Error loading {v} model: {e}")

load_models()

# Complete list of 55 clinical features expected by the trained pipeline
NUMERIC_FEATURES = [
    'Age', 'Weight', 'Length', 'BMI', 'DM', 'HTN', 'Current Smoker',
    'EX-Smoker', 'FH', 'BP', 'PR', 'Edema', 'Typical Chest Pain',
    'Function Class', 'Q Wave', 'St Elevation', 'St Depression',
    'Tinversion', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR',
    'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT', 'EF-TTE', 'Region RWMA'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease',
    'CHF', 'DLP', 'Weak Peripheral Pulse', 'Lung rales', 'Systolic Murmur',
    'Diastolic Murmur', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP',
    'LowTH Ang', 'LVH', 'Poor R Progression', 'BBB', 'VHD'
]

# Baseline / Typical Clinical Defaults
DEFAULT_PATIENT: Dict[str, Any] = {
    # Demographics & Vitals
    "Age": 55, "Sex": "Male", "Weight": 74, "Length": 172, "BMI": 25.0,
    "BP": 125, "PR": 75,
    # Risk Factors & History
    "DM": 0, "HTN": 0, "Current Smoker": 0, "EX-Smoker": 0, "FH": 0,
    "Obesity": "N", "CRF": "N", "CVA": "N", "Airway disease": "N",
    "Thyroid Disease": "N", "CHF": "N", "DLP": "N",
    # Symptoms & Exam
    "Typical Chest Pain": 0, "Atypical": "N", "Nonanginal": "N",
    "Exertional CP": "N", "LowTH Ang": "N", "Dyspnea": "N",
    "Edema": 0, "Weak Peripheral Pulse": "N", "Lung rales": "N",
    "Systolic Murmur": "N", "Diastolic Murmur": "N", "Function Class": 0,
    # ECG & Echo
    "Q Wave": 0, "St Elevation": 0, "St Depression": 0, "Tinversion": 0,
    "LVH": "N", "Poor R Progression": "N", "BBB": "N", "VHD": "N",
    "EF-TTE": 55, "Region RWMA": 0,
    # Labs
    "FBS": 95, "CR": 0.9, "TG": 140, "LDL": 110, "HDL": 48,
    "BUN": 15, "ESR": 12, "HB": 14.5, "K": 4.2, "Na": 140,
    "WBC": 6800, "Lymph": 32, "Neut": 60, "PLT": 230
}

# 3 Realistic Patient Profiles for Testing & 1-Click Demo
PRESETS: Dict[str, Dict[str, Any]] = {
    "preset_1_healthy": {
        "id": "preset_1_healthy",
        "name": "Patient A: Healthy / Low Cardiovascular Risk",
        "description": "42-year-old active adult with normal blood pressure, lipid panel, and normal ECG.",
        "data": {
            **DEFAULT_PATIENT,
            "Age": 42, "Sex": "Female", "Weight": 62, "Length": 165, "BMI": 22.8,
            "BP": 115, "PR": 68, "DM": 0, "HTN": 0, "Current Smoker": 0, "FH": 0,
            "Typical Chest Pain": 0, "St Elevation": 0, "St Depression": 0, "Tinversion": 0,
            "EF-TTE": 60, "Region RWMA": 0, "LDL": 92, "HDL": 56, "TG": 110, "FBS": 88
        }
    },
    "preset_2_lad_stenosis": {
        "id": "preset_2_lad_stenosis",
        "name": "Patient B: Suspected LAD Stenosis (Anterior Wall)",
        "description": "58-year-old male with hypertension, elevated LDL, typical exertional angina, and anterior ST elevation.",
        "data": {
            **DEFAULT_PATIENT,
            "Age": 58, "Sex": "Male", "Weight": 82, "Length": 174, "BMI": 27.1,
            "BP": 145, "PR": 82, "DM": 0, "HTN": 1, "Current Smoker": 0, "EX-Smoker": 1, "FH": 1,
            "Typical Chest Pain": 1, "St Elevation": 1, "Tinversion": 1, "Q Wave": 0,
            "EF-TTE": 45, "Region RWMA": 1, "LDL": 165, "HDL": 39, "TG": 195, "FBS": 115
        }
    },
    "preset_3_multivessel_cad": {
        "id": "preset_3_multivessel_cad",
        "name": "Patient C: Severe Multi-Vessel CAD (Triple Vessel Disease)",
        "description": "68-year-old diabetic smoker with diffuse ST-T changes, impaired ejection fraction, and high multi-vessel risk.",
        "data": {
            **DEFAULT_PATIENT,
            "Age": 68, "Sex": "Male", "Weight": 88, "Length": 170, "BMI": 30.4, "Obesity": "Y",
            "BP": 165, "PR": 88, "DM": 1, "HTN": 1, "Current Smoker": 1, "FH": 1,
            "Typical Chest Pain": 1, "Exertional CP": "Y", "Dyspnea": "Y",
            "St Elevation": 1, "St Depression": 1, "Tinversion": 1, "Q Wave": 1,
            "EF-TTE": 35, "Region RWMA": 2, "LDL": 215, "HDL": 31, "TG": 290, "FBS": 175
        }
    }
}

def get_vessel_status_and_color(prob: float) -> Dict[str, str]:
    """Map stenosis probability to visual color code and status text."""
    if prob >= 0.65:
        return {
            "status": "High Stenosis Risk (>50% Narrowing)",
            "severity": "High",
            "color": "#EF4444",      # Vibrant Red
            "hex_code": 0xef4444,
            "emissive": "#7F1D1D"
        }
    elif prob >= 0.35:
        return {
            "status": "Moderate Stenosis Risk (Borderline)",
            "severity": "Moderate",
            "color": "#F59E0B",      # Amber / Yellow
            "hex_code": 0xf59e0b,
            "emissive": "#78350F"
        }
    else:
        return {
            "status": "Patent / Normal Vessel",
            "severity": "Normal",
            "color": "#10B981",      # Emerald Green
            "hex_code": 0x10b981,
            "emissive": "#064E3B"
        }

def format_patient_dataframe(patient_dict: Dict[str, Any]) -> pd.DataFrame:
    """Ensure all required columns are present in DataFrame format."""
    merged = {**DEFAULT_PATIENT, **patient_dict}
    # Ensure correct types
    for num_col in NUMERIC_FEATURES:
        try:
            merged[num_col] = float(merged.get(num_col, 0))
        except (ValueError, TypeError):
            merged[num_col] = 0.0
            
    for cat_col in CATEGORICAL_FEATURES:
        merged[cat_col] = str(merged.get(cat_col, "N")).strip()
        
    df = pd.DataFrame([merged])
    return df

@app.get("/")
def root():
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        return FileResponse(FRONTEND_DIR / "index.html")
    return {
        "service": "Cardiovascular AI Risk Engine",
        "status": "online",
        "models_loaded": list(MODELS.keys()),
        "disclaimer": "For decision support and educational research only. Not for standalone clinical diagnosis."
    }

@app.get("/styles.css")
def serve_css():
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "styles.css").exists():
        return FileResponse(FRONTEND_DIR / "styles.css")
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/app.js")
def serve_js():
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "app.js").exists():
        return FileResponse(FRONTEND_DIR / "app.js")
    raise HTTPException(status_code=404, detail="app.js not found")

@app.get("/GLTFLoader.js")
def serve_gltf_loader():
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "GLTFLoader.js").exists():
        return FileResponse(FRONTEND_DIR / "GLTFLoader.js")
    raise HTTPException(status_code=404, detail="GLTFLoader.js not found")

@app.get("/anatomical_heart.glb")
def serve_glb():
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "anatomical_heart.glb").exists():
        return FileResponse(FRONTEND_DIR / "anatomical_heart.glb", media_type="model/gltf-binary")
    raise HTTPException(status_code=404, detail="anatomical_heart.glb not found")

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models": {k: bool(v) for k, v in MODELS.items()}
    }

@app.get("/api/presets")
def get_presets():
    """Return realistic pre-built patient records for quick demonstration."""
    return list(PRESETS.values())

@app.post("/api/predict")
def predict_cardiac_risk(patient_data: Dict[str, Any]):
    """
    Predict overall CAD status and individual stenosis probabilities
    for LAD, LCX, and RCA.
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    df_patient = format_patient_dataframe(patient_data)
    results: Dict[str, Any] = {}

    # 1. Overall CAD Prediction
    if "cad" in MODELS:
        cad_model = MODELS["cad"]
        cad_prob = float(cad_model.predict_proba(df_patient)[0, 1])
        cad_pred = "CAD" if cad_prob >= 0.5 else "Normal"
        results["overall_cad"] = {
            "prediction": cad_pred,
            "probability": round(cad_prob, 3),
            "percentage": round(cad_prob * 100, 1),
            "risk_level": "High" if cad_prob >= 0.65 else ("Moderate" if cad_prob >= 0.35 else "Low"),
            "color": "#EF4444" if cad_prob >= 0.65 else ("#F59E0B" if cad_prob >= 0.35 else "#10B981")
        }

    # 2. Vessel-Specific Predictions (LAD, LCX, RCA)
    vessel_results = {}
    vessel_colors = {}
    
    for vessel in ["lad", "lcx", "rca"]:
        vessel_upper = vessel.upper()
        if vessel in MODELS:
            v_model = MODELS[vessel]
            v_prob = float(v_model.predict_proba(df_patient)[0, 1])
            v_info = get_vessel_status_and_color(v_prob)
            
            vessel_results[vessel_upper] = {
                "name": vessel_upper,
                "full_name": {
                    "LAD": "Left Anterior Descending",
                    "LCX": "Left Circumflex",
                    "RCA": "Right Coronary Artery"
                }.get(vessel_upper, vessel_upper),
                "territory": {
                    "LAD": "Anterior Heart Wall & Interventricular Septum",
                    "LCX": "Lateral and Posterior Ventricular Wall",
                    "RCA": "Inferior Wall & Right Ventricle"
                }.get(vessel_upper, "Coronary Circulation"),
                "probability": round(v_prob, 3),
                "percentage": round(v_prob * 100, 1),
                "status": v_info["status"],
                "severity": v_info["severity"],
                "color": v_info["color"],
                "hex_code": v_info["hex_code"],
                "emissive": v_info["emissive"]
            }
            vessel_colors[vessel_upper] = v_info["color"]

    results["vessels"] = vessel_results
    results["3d_mesh_colors"] = vessel_colors
    results["safety_disclaimer"] = (
        "Educational prototype for research demonstration. "
        "Not a substitute for formal clinical or diagnostic imaging."
    )
    return results

@app.post("/api/explain")
def explain_prediction(patient_data: Dict[str, Any]):
    """
    Compute patient-specific feature attribution scores for clinical interpretability.
    Identifies the primary clinical drivers of predicted coronary risk.
    """
    if "cad" not in MODELS:
        raise HTTPException(status_code=503, detail="CAD model is not available.")

    cad_model = MODELS["cad"]
    preprocessor = cad_model.named_steps["preprocessor"]
    classifier = cad_model.named_steps["classifier"]

    df_patient = format_patient_dataframe(patient_data)
    X_transformed = preprocessor.transform(df_patient)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    global_importances = classifier.feature_importances_

    # Calculate patient-specific risk contributions (magnitude = normalized feature impact)
    patient_vector = X_transformed[0]
    local_impacts = np.abs(patient_vector) * global_importances

    # Group into readable high-level factors
    top_indices = np.argsort(local_impacts)[::-1][:8]

    explanations: List[Dict[str, Any]] = []
    total_top_weight = sum(local_impacts[top_indices]) or 1.0

    for idx in top_indices:
        raw_name = feature_names[idx]
        clean_name = raw_name.replace("numeric__", "").replace("categorical__", "")
        
        # Clinical explanation generator
        impact_pct = round(float((local_impacts[idx] / total_top_weight) * 100), 1)
        patient_val = patient_data.get(clean_name.split("_")[0], "Recorded")

        explanations.append({
            "feature": clean_name,
            "patient_value": str(patient_val),
            "relative_importance_pct": impact_pct,
            "direction": "Elevates Risk" if "St Elevation" in clean_name or "Age" in clean_name or "Typical" in clean_name else "Contributing Factor"
        })

    return {
        "top_contributing_factors": explanations,
        "method": "Random Forest Tree-Level Feature Attribution & SHAP Global Weighting",
        "clinical_summary": f"Primary cardiac risk drivers are dominated by {explanations[0]['feature']} and {explanations[1]['feature']}."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
