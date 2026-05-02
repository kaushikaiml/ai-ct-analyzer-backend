from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
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

MEDICAL_DATA = {
    "Adenocarcinoma": {
        "description": "Adenocarcinoma is the most common type of lung cancer, arising from glandular cells in the peripheral lung tissue. It is frequently associated with non-smokers and women.",
        "symptoms": [
            "Persistent dry cough (>3 weeks)",
            "Chest pain (worsens with deep breathing)",
            "Progressive shortness of breath",
            "Unexplained weight loss (>10% body weight)",
            "Chronic fatigue and weakness",
            "Hemoptysis (blood in sputum)",
            "Recurrent respiratory infections",
            "Digital clubbing (in advanced stages)"
        ],
        "risk_factors": [
            "Non-smokers at higher relative risk",
            "Exposure to radon gas",
            "Air pollution exposure",
            "Family history of lung cancer",
            "Exposure to asbestos or carcinogens",
            "EGFR/ALK gene mutations"
        ],
        "findings": [
            "Peripheral lung mass identified — right/left lobe",
            "Irregular spiculated margins observed",
            "Ground-glass opacity pattern detected",
            "Size estimation: 2-4 cm (requires CT measurement)",
            "Possible pleural involvement noted",
            "Mediastinal lymph node enlargement suspected",
            "No definitive cavitation observed",
            "Vascular invasion cannot be excluded"
        ],
        "staging": {
            "probable_stage": "Stage II - III (based on imaging)",
            "tnm": "T2-T3, N1-N2, M0 (estimated)",
            "note": "Pathological staging requires biopsy confirmation"
        },
        "treatment": [
            "Surgical resection (Lobectomy/Segmentectomy) — if resectable",
            "Platinum-based Chemotherapy (Cisplatin + Pemetrexed)",
            "Targeted Therapy — EGFR inhibitors (Erlotinib/Osimertinib)",
            "ALK inhibitors if ALK rearrangement confirmed",
            "Immunotherapy — PD-L1 inhibitors (Pembrolizumab)",
            "Stereotactic Body Radiation Therapy (SBRT)",
            "Concurrent Chemoradiation (if unresectable)",
            "Palliative care for symptom management"
        ],
        "specialist_referral": [
            "Pulmonologist — for bronchoscopy and lung function",
            "Thoracic Surgeon — for surgical evaluation",
            "Medical Oncologist — for chemotherapy planning",
            "Radiation Oncologist — for radiotherapy assessment",
            "Pathologist — for biopsy and molecular testing"
        ],
        "additional_tests": [
            "CT-guided biopsy for histopathological confirmation",
            "PET-CT scan for metastasis detection",
            "MRI Brain — rule out brain metastasis",
            "Pulmonary Function Tests (PFT)",
            "EGFR/ALK/ROS1 molecular profiling",
            "Complete Blood Count (CBC) and LFT",
            "Bone scan if bone metastasis suspected"
        ],
        "followup_plan": [
            "Immediate oncology consultation within 1 week",
            "Multidisciplinary Tumor Board review",
            "CT chest every 3 months post-treatment",
            "Annual PET scan for surveillance",
            "Pulmonary rehabilitation program",
            "Psychological counseling and support"
        ],
        "severity": "High",
        "survival_rate": "25-30% (5-year overall survival)",
        "prognosis": "Guarded — depends on stage, molecular profile, and treatment response"
    },

    "Large Cell Carcinoma": {
        "description": "Large Cell Carcinoma is an aggressive, undifferentiated form of non-small cell lung cancer. It tends to grow rapidly and can occur in any part of the lung, often presenting at an advanced stage.",
        "symptoms": [
            "Severe persistent chest pain",
            "Rapid unexplained weight loss",
            "Productive cough with blood-tinged sputum",
            "Progressive dyspnea at rest",
            "Superior vena cava syndrome",
            "Dysphagia (difficulty swallowing)",
            "Paraneoplastic syndromes",
            "Bone pain if metastasis present"
        ],
        "risk_factors": [
            "Heavy long-term smoking history",
            "Occupational carcinogen exposure",
            "Radiation exposure history",
            "Chronic obstructive pulmonary disease (COPD)",
            "HIV or immunosuppression",
            "Prior lung malignancy"
        ],
        "findings": [
            "Large peripheral/central mass identified",
            "Necrotic center with cavitation noted",
            "Mediastinal widening observed",
            "Possible chest wall invasion",
            "Hilar lymphadenopathy detected",
            "Pleural effusion may be present",
            "Satellite nodules cannot be excluded",
            "Vascular encasement suspected"
        ],
        "staging": {
            "probable_stage": "Stage III - IV (based on imaging)",
            "tnm": "T3-T4, N2-N3, M0-M1 (estimated)",
            "note": "Requires urgent pathological staging"
        },
        "treatment": [
            "Platinum-based Chemotherapy (first-line)",
            "Nivolumab or Pembrolizumab (Immunotherapy)",
            "Chemoimmunotherapy combination",
            "Palliative Radiation for symptom control",
            "Surgery only if early-stage and resectable",
            "Bevacizumab (anti-angiogenic therapy)",
            "Clinical trial enrollment recommended",
            "Best supportive/palliative care"
        ],
        "specialist_referral": [
            "Medical Oncologist — urgent consultation",
            "Thoracic Surgeon — resectability assessment",
            "Radiation Oncologist — palliative/curative RT",
            "Palliative Care Specialist",
            "Interventional Pulmonologist"
        ],
        "additional_tests": [
            "Urgent CT-guided or bronchoscopic biopsy",
            "Whole-body PET-CT scan",
            "MRI Brain with contrast",
            "Bone marrow biopsy if indicated",
            "Comprehensive molecular profiling",
            "Serum tumor markers (CEA, NSE)",
            "Echocardiogram if pericardial involvement"
        ],
        "followup_plan": [
            "Urgent oncology referral within 48-72 hours",
            "Tumor Board discussion within 1 week",
            "Response assessment CT after 2 cycles chemo",
            "Monthly clinical review during treatment",
            "Palliative care integration from diagnosis",
            "Family counseling and support services"
        ],
        "severity": "Very High",
        "survival_rate": "15-20% (5-year overall survival)",
        "prognosis": "Poor — aggressive tumor with rapid progression"
    },

    "Normal": {
        "description": "No significant pulmonary malignancy or structural abnormality detected on CT imaging. Lung parenchyma appears within normal limits. Continued preventive health monitoring is advised.",
        "symptoms": [
            "No active alarming symptoms detected",
            "Routine health screening recommended",
            "Monitor for new onset respiratory symptoms",
            "Preventive lifestyle measures advised"
        ],
        "risk_factors": [
            "Smoking history (if applicable)",
            "Age above 50 years",
            "Family history of lung cancer",
            "Occupational exposure to chemicals",
            "Radon exposure in living environment"
        ],
        "findings": [
            "No pulmonary mass or nodule identified",
            "Normal lung parenchymal architecture",
            "No pleural effusion or thickening",
            "Mediastinum appears normal in width",
            "No significant lymphadenopathy",
            "Airways appear patent bilaterally",
            "No bony erosion or rib abnormality",
            "Heart size within normal limits"
        ],
        "staging": {
            "probable_stage": "Not Applicable",
            "tnm": "T0, N0, M0",
            "note": "No malignancy detected on current imaging"
        },
        "treatment": [
            "No immediate treatment required",
            "Annual low-dose CT screening (if high risk)",
            "Smoking cessation program (if smoker)",
            "Regular physical activity — 150 min/week",
            "Balanced diet rich in antioxidants",
            "Avoid occupational carcinogen exposure",
            "Maintain healthy BMI",
            "Annual comprehensive health checkup"
        ],
        "specialist_referral": [
            "General Physician — for routine monitoring",
            "Pulmonologist — if symptoms develop",
            "Preventive Medicine Specialist",
            "Nutritionist for lifestyle guidance"
        ],
        "additional_tests": [
            "Annual chest X-ray for baseline",
            "Pulmonary Function Tests (PFT) if symptomatic",
            "CBC and metabolic panel annually",
            "Sputum cytology if heavy smoker",
            "Low-dose CT screening annually (>55 years, smokers)"
        ],
        "followup_plan": [
            "Routine follow-up in 12 months",
            "Immediate review if new symptoms appear",
            "Annual cancer screening program enrollment",
            "Lifestyle modification counseling",
            "Vaccination (Influenza, Pneumococcal)",
            "Mental health and wellness assessment"
        ],
        "severity": "Normal",
        "survival_rate": "Normal life expectancy",
        "prognosis": "Excellent — no malignancy detected"
    },

    "Squamous Cell Carcinoma": {
        "description": "Squamous Cell Carcinoma arises from the flat cells lining the airways (bronchi), typically in the central lung regions. It is strongly associated with cigarette smoking and tends to grow slower than other types.",
        "symptoms": [
            "Chronic productive cough with purulent sputum",
            "Frank hemoptysis (bright red blood)",
            "Progressive wheezing and stridor",
            "Central chest pain and pressure",
            "Recurrent post-obstructive pneumonia",
            "Hoarseness of voice (recurrent laryngeal nerve involvement)",
            "Dysphagia if esophageal compression",
            "Pancoast syndrome (if superior sulcus)"
        ],
        "risk_factors": [
            "Heavy cigarette smoking (primary risk factor)",
            "Passive smoke exposure",
            "Occupational exposure (asbestos, silica)",
            "Chronic bronchitis/COPD",
            "HPV infection (rare association)",
            "Vitamin A deficiency"
        ],
        "findings": [
            "Central airway mass with endobronchial component",
            "Post-obstructive atelectasis/consolidation",
            "Possible cavitation with thick irregular walls",
            "Hilar and mediastinal lymphadenopathy",
            "Bronchial wall thickening noted",
            "No ground-glass opacity pattern",
            "Possible direct chest wall extension",
            "Ipsilateral pleural thickening"
        ],
        "staging": {
            "probable_stage": "Stage II - III (based on imaging)",
            "tnm": "T2-T3, N1-N2, M0 (estimated)",
            "note": "Central location requires bronchoscopic staging"
        },
        "treatment": [
            "Surgical resection (Pneumonectomy/Lobectomy)",
            "Concurrent Chemoradiation (Cisplatin + Radiation)",
            "Bronchoscopic intervention for airway obstruction",
            "Necitumumab (EGFR antibody) + Chemotherapy",
            "PD-1/PD-L1 Immunotherapy (Pembrolizumab)",
            "Endobronchial brachytherapy for palliation",
            "Photodynamic therapy for early central lesions",
            "Nutritional support and pulmonary rehabilitation"
        ],
        "specialist_referral": [
            "Thoracic Surgeon — surgical candidacy evaluation",
            "Interventional Pulmonologist — bronchoscopy",
            "Medical Oncologist — systemic therapy",
            "Radiation Oncologist — RT planning",
            "ENT Specialist — if voice involvement"
        ],
        "additional_tests": [
            "Flexible bronchoscopy with biopsy",
            "EBUS (Endobronchial Ultrasound) for staging",
            "PET-CT for nodal and distant staging",
            "MRI spine if neurological symptoms",
            "Pulmonary Function Tests pre-surgery",
            "PD-L1 expression testing",
            "Serum SCC antigen levels"
        ],
        "followup_plan": [
            "Oncology consultation within 5-7 days",
            "Multidisciplinary team review",
            "CT chest every 3 months for 2 years",
            "Bronchoscopy follow-up post-treatment",
            "Smoking cessation program — mandatory",
            "Pulmonary rehabilitation post-surgery",
            "Psychosocial support and counseling"
        ],
        "severity": "High",
        "survival_rate": "20-25% (5-year overall survival)",
        "prognosis": "Moderate — better response to RT than adenocarcinoma"
    }
}

class CTScanCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

model = CTScanCNN()
model.load_state_dict(torch.load("ct_model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

@app.get("/")
def home():
    return {"message": "AI CT Scan Analyzer Ready!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output).item()
        confidence = torch.softmax(output, dim=1)[0][prediction].item()

    diagnosis = CLASSES[prediction]
    data = MEDICAL_DATA[diagnosis]

    return {
        "diagnosis": diagnosis,
        "confidence": f"{confidence*100:.1f}%",
        "description": data["description"],
        "severity": data["severity"],
        "survival_rate": data["survival_rate"],
        "prognosis": data["prognosis"],
        "symptoms": data["symptoms"],
        "risk_factors": data["risk_factors"],
        "findings": data["findings"],
        "staging": data["staging"],
        "treatment": data["treatment"],
        "specialist_referral": data["specialist_referral"],
        "additional_tests": data["additional_tests"],
        "followup_plan": data["followup_plan"],
        "report_date": datetime.now().strftime("%d %B %Y, %H:%M")
    }