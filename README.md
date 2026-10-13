# CardioVision 3D — AI Multi-Vessel Cardiovascular Risk Visualization & Prediction

> **Interactive 3D Anatomical Risk Mapping & Explainable AI for Coronary Artery Disease (CAD)**  
> Developed for the Hackathon Problem Statement: *Cardiovascular Risk Visualization & Prediction*

---

## 🫀 Project Overview

Cardiovascular disease remains the leading cause of mortality worldwide. Standard statistical risk calculators generate abstract percentages that fail to convey **where** and **how** severe arterial pathology is developing inside a patient's coronary circulation.

**CardioVision 3D** bridges this critical gap by translating clinical, ECG, laboratory, and echocardiographic biomarkers into an **interactive 3D anatomical heart model** with localized coronary artery stenosis mapping and patient-specific explainable AI (XAI).

---

## 🚀 Key Features

### 1. Multi-Target Machine Learning Pipeline
* **Overall CAD Classification:** Predicts general CAD status using Random Forest ensemble models.
* **Vessel-Level Stenosis Models:** Dedicated models predicting localized significant stenosis ($>50\%$ occlusion) for the three primary coronary arteries:
  * **LAD (Left Anterior Descending):** Supplies the anterior myocardial wall and interventricular septum.
  * **LCX (Left Circumflex):** Supplies the lateral and posterior left ventricle.
  * **RCA (Right Coronary Artery):** Supplies the right ventricle and inferior diaphragmatic wall.
* **Leakage-Free Validation:** Target columns (`Cath`, `LAD`, `LCX`, `RCA`) are strictly excluded from input training features. Evaluated using 5-fold Stratified Cross-Validation across Accuracy, Precision, Recall, F1, and ROC-AUC.

### 2. Interactive 3D Spatial Visual Mapping (Three.js & WebGL)
* **Real-Time Anatomical Rendering:** Procedurally sculpted 3D heart with myocardium, ascending aorta, pulmonary trunk, and anatomically routed coronary arterial trees (including diagonal and marginal branches).
* **Dynamic Color-Coded Perfusion:**
  * 🟢 **Patent / Normal ($<35\%$ risk):** `#10B981` (Emerald Green)
  * 🟡 **Borderline Stenosis ($35\% - 65\%$ risk):** `#F59E0B` (Amber / Yellow)
  * 🔴 **Severe Stenosis ($>65\%$ risk):** `#EF4444` (Vibrant Red with animated emissive pulsation)
* **3D Raycasting & Inspection:** Hover or click directly on any 3D coronary artery to inspect vessel name, localized perfusion territory, and predicted risk score.
* **Camera Views:** Instant camera orientation presets for **Anterior (LAD)**, **Lateral (LCX)**, **Inferior (RCA)**, and **Full Overview**.
* **Physiological Cardiac Pulsation:** Toggleable systolic/diastolic heartbeat pulsation.

### 3. Clinical Intelligence & Explainability Dashboard
* **Patient Presets (1-Click Demo):**
  * *Patient A:* Healthy / Low Risk adult.
  * *Patient B:* Suspected LAD Stenosis (Anterior wall ischemia).
  * *Patient C:* Severe Multi-Vessel CAD (Triple Vessel Disease).
* **Custom Clinical Simulation Modal:** Interactive form allowing clinicians to adjust vitals (Age, BP, Heart Rate, Echo Ejection Fraction, ST Elevation, Lipids, etc.) and observe live 3D arterial color updates.
* **Tree-Level SHAP Feature Attribution:** Dynamic breakdown displaying the exact physiological biomarkers driving the patient's cardiac risk.

### 4. Mandatory Clinical Safety Disclaimer
Prominently integrated into the UI header and API response:
> *⚠️ Clinical Decision Support & Educational Research Prototype. Not a substitute for formal diagnostic coronary angiography or diagnostic imaging.*

---

## 🛠️ Architecture & Tech Stack

```
Cardiovascular-AI-project/
├── backend/
│   └── app.py                     # FastAPI backend serving predictions, SHAP, and presets
├── frontend/
│   ├── index.html                 # Modern glassmorphism medical dashboard UI
│   ├── app.js                     # Three.js 3D Heart Engine, OrbitControls, and API sync
│   └── styles.css                 # HUD styling, animations, and responsive layouts
├── ml/
│   ├── train_model.py             # Overall CAD Random Forest training script
│   ├── train_vessel_models.py      # LAD, LCX, RCA multi-vessel training pipeline
│   ├── explain_vessel_models.py   # Global SHAP feature importance visualizer
│   └── cross_validate_vessels.py  # 5-fold Stratified Cross-Validation runner
├── models/                        # Pre-trained scikit-learn model weights (.joblib)
│   ├── cad_model.joblib
│   ├── lad_model.joblib
│   ├── lcx_model.joblib
│   └── rca_model.joblib
├── shap_results/                  # Global feature importance summary plots
├── run_backend.bat                # 1-Click launcher script
└── README.md
```

* **Frontend:** Three.js (WebGL), OrbitControls, Tailwind CSS, Lucide Icons.
* **Backend:** FastAPI, Uvicorn, Pydantic, Starlette.
* **Machine Learning:** Scikit-Learn (Random Forest, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder), Joblib, NumPy, Pandas.
* **Dataset:** UCI Extension of Z-Alizadeh Sani Cardiovascular Dataset (303 patient clinical records, 55 diagnostic features).

---

## ⚡ Quick Start & Running the Prototype

### Prerequisites
* Python 3.10+ installed.

### 1. Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas openpyxl joblib
```

### 2. Launch the Application
Run the launcher batch file or start via terminal:
```bash
# Option A: Windows 1-click
run_backend.bat

# Option B: Manual command
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open in Browser
Visit **`http://127.0.0.1:8000`** in your web browser.

---

## 📊 Evaluation & Validation Metrics

| Model Target | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall CAD** | **84.2%** | **88.9%** | **88.9%** | **88.9%** | **0.931** |
| **LAD Stenosis** | **80.3%** | **82.5%** | **84.6%** | **83.5%** | **0.884** |
| **LCX Stenosis** | **76.3%** | **70.6%** | **63.2%** | **66.7%** | **0.842** |
| **RCA Stenosis** | **78.9%** | **75.0%** | **64.3%** | **69.2%** | **0.851** |

*Validated using 5-Fold Stratified Cross-Validation and held-out 25% test partition without target leakage.*

---

## 👥 Hackathon Team & Acknowledgements
* **Dataset:** UCI Machine Learning Repository / Z-Alizadeh Sani Dataset.
* **License:** Educational / Open Source for Research Demonstration.