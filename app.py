from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("ct_model.onnx")

CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

DISEASE_DATA = {
    "Adenocarcinoma": {
        "severity": "High",
        "survival_rate": "25-30% (5-year)",
        "description": "Adenocarcinoma is the most common type of lung cancer, arising from glandular cells in the lung periphery.",
        "prognosis": "Prognosis depends on stage at diagnosis. Early detection significantly improves outcomes. Targeted therapy available for EGFR/ALK mutations.",
        "staging": {
            "probable_stage": "Stage II-III (based on imaging)",
            "tnm": "T2-T3, N1-N2, M0",
            "note": "Definitive staging requires biopsy and PET-CT scan"
        },
        "findings": [
            "Peripheral lung mass detected",
            "Irregular spiculated margins observed",
            "Possible pleural involvement",
            "Ground-glass opacity pattern present"
        ],
        "symptoms": ["Persistent cough", "Dyspnea", "Hemoptysis", "Chest pain", "Weight loss", "Fatigue"],
        "risk_factors": ["Smoking history", "Radon exposure", "Air pollution", "Family history", "Asbestos exposure"],
        "treatment": [
            "Surgery (Lobectomy if resectable)",
            "Chemotherapy (Carboplatin + Paclitaxel)",
            "Targeted therapy (EGFR inhibitors if mutation+)",
            "Immunotherapy (Pembrolizumab)"
        ],
        "specialist_referral": [
            "Thoracic Oncologist — Primary management",
            "Cardiothoracic Surgeon — Surgical evaluation",
            "Radiation Oncologist — If surgery not feasible"
        ],
        "additional_tests": [
            "PET-CT Scan for staging",
            "Bronchoscopy with biopsy",
            "EGFR/ALK/ROS1 molecular testing",
            "Pulmonary function tests"
        ],
        "followup_plan": [
            "CT scan every 3 months for first year",
            "Oncology review every 6 weeks during treatment",
            "Annual CT scan after remission"
        ]
    },
    "Large Cell Carcinoma": {
        "severity": "Very High",
        "survival_rate": "15-20% (5-year)",
        "description": "Large cell carcinoma is an aggressive undifferentiated lung cancer that tends to grow and spread quickly.",
        "prognosis": "Poor prognosis due to aggressive nature and tendency for early metastasis. Multidisciplinary treatment approach required.",
        "staging": {
            "probable_stage": "Stage III-IV (based on imaging)",
            "tnm": "T3-T4, N2-N3, M0-M1",
            "note": "Often diagnosed at advanced stage. Biopsy essential for confirmation."
        },
        "findings": [
            "Large central or peripheral mass",
            "Poorly defined tumor margins",
            "Mediastinal lymph node enlargement",
            "Possible pleural effusion"
        ],
        "symptoms": ["Severe dyspnea", "Persistent cough", "Chest pain", "Hemoptysis", "Significant weight loss", "Night sweats"],
        "risk_factors": ["Heavy smoking", "Occupational carcinogens", "Radiation exposure", "Genetic predisposition"],
        "treatment": [
            "Combination chemotherapy (Cisplatin + Etoposide)",
            "Immunotherapy (Nivolumab/Pembrolizumab)",
            "Radiation therapy",
            "Palliative care if Stage IV"
        ],
        "specialist_referral": [
            "Thoracic Oncologist — Urgent evaluation",
            "Interventional Radiologist — Biopsy",
            "Palliative Care Team — Symptom management"
        ],
        "additional_tests": [
            "Urgent PET-CT scan",
            "CT-guided biopsy",
            "Brain MRI to rule out metastasis",
            "Complete blood count and liver function tests"
        ],
        "followup_plan": [
            "Urgent oncology consultation within 1 week",
            "CT scan every 6-8 weeks during treatment",
            "Monthly symptom assessment"
        ]
    },
    "Normal": {
        "severity": "Normal",
        "survival_rate": "99%+ (5-year)",
        "description": "No significant abnormality detected in the CT scan. Lung fields appear clear with no suspicious masses or nodules.",
        "prognosis": "Excellent. No evidence of malignancy. Routine health screening recommended.",
        "staging": {
            "probable_stage": "N/A — No cancer detected",
            "tnm": "T0, N0, M0",
            "note": "Regular annual screening recommended especially for high-risk individuals"
        },
        "findings": [
            "Lung fields appear clear",
            "No suspicious nodules or masses",
            "Normal bronchovascular markings",
            "No pleural effusion detected"
        ],
        "symptoms": ["No significant symptoms detected", "Routine screening advised"],
        "risk_factors": ["Smoking (if applicable)", "Age > 50", "Family history of lung cancer"],
        "treatment": [
            "No treatment required",
            "Annual low-dose CT screening if high-risk",
            "Smoking cessation if applicable",
            "Regular health checkups"
        ],
        "specialist_referral": [
            "General Physician — Routine annual checkup",
            "Pulmonologist — If respiratory symptoms develop"
        ],
        "additional_tests": [
            "Annual CT screening if high-risk",
            "Pulmonary function test if symptomatic",
            "Chest X-ray as baseline"
        ],
        "followup_plan": [
            "Annual CT screening recommended",
            "Report any new respiratory symptoms immediately",
            "Maintain healthy lifestyle and avoid smoking"
        ]
    },
    "Squamous Cell Carcinoma": {
        "severity": "High",
        "survival_rate": "20-25% (5-year)",
        "description": "Squamous cell carcinoma arises from the flat cells lining the airways, typically in the central bronchi.",
        "prognosis": "Moderate to poor prognosis. Strongly associated with smoking. Surgery may be curative if detected early.",
        "staging": {
            "probable_stage": "Stage II-III (based on imaging)",
            "tnm": "T2-T3, N1-N2, M0",
            "note": "Central location may complicate surgical resection. Bronchoscopy essential."
        },
        "findings": [
            "Central hilar mass detected",
            "Bronchial obstruction pattern",
            "Post-obstructive atelectasis present",
            "Cavitation may be present"
        ],
        "symptoms": ["Chronic cough", "Hemoptysis", "Wheezing", "Recurrent pneumonia", "Chest pain", "Hoarseness"],
        "risk_factors": ["Heavy smoking (primary risk)", "Occupational dust exposure", "Air pollution", "Chronic lung disease"],
        "treatment": [
            "Surgery (if resectable)",
            "Concurrent chemoradiation",
            "Chemotherapy (Cisplatin + Gemcitabine)",
            "Immunotherapy (Pembrolizumab for PD-L1+)"
        ],
        "specialist_referral": [
            "Thoracic Oncologist — Primary management",
            "Pulmonologist — Bronchoscopy evaluation",
            "Radiation Oncologist — Chemoradiation planning"
        ],
        "additional_tests": [
            "Bronchoscopy with biopsy",
            "PET-CT for staging",
            "PD-L1 expression testing",
            "Pulmonary function tests pre-surgery"
        ],
        "followup_plan": [
            "CT scan every 3 months during treatment",
            "Smoking cessation program — mandatory",
            "Oncology review every 4-6 weeks"
        ]
    }
}

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return arr.transpose(2, 0, 1)[np.newaxis, :]

@app.get("/")
def root():
    return {"status": "AI CT Analyzer Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    probs = outputs[0][0]

    idx = int(np.argmax(probs))
    diagnosis = CLASSES[idx]
    confidence = f"{float(probs[idx]) * 100:.1f}%"

    data = DISEASE_DATA[diagnosis]

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "severity": data["severity"],
        "survival_rate": data["survival_rate"],
        "description": data["description"],
        "prognosis": data["prognosis"],
        "staging": data["staging"],
        "findings": data["findings"],
        "symptoms": data["symptoms"],
        "risk_factors": data["risk_factors"],
        "treatment": data["treatment"],
        "specialist_referral": data["specialist_referral"],
        "additional_tests": data["additional_tests"],
        "followup_plan": data["followup_plan"],
        "report_date": datetime.now().strftime("%d %B %Y, %H:%M"),
    }