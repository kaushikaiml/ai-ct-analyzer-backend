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

# ONNX model load karo
session = ort.InferenceSession("ct_model.onnx")

CLASSES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma", 
    "Normal",
    "Squamous Cell Carcinoma"
]

MEDICAL_DATA = {
    "Adenocarcinoma": {
        "severity": "High",
        "survival_rate": "25-30% (5-year)",
        "symptoms": ["Persistent cough", "Chest pain", "Shortness of breath", "Weight loss"],
        "treatment": ["Surgery", "Chemotherapy", "Targeted therapy", "Immunotherapy"],
        "findings": ["Peripheral lung mass detected", "Irregular margins", "Possible pleural involvement"]
    },
    "Large Cell Carcinoma": {
        "severity": "Very High",
        "survival_rate": "15-20% (5-year)",
        "symptoms": ["Severe chest pain", "Rapid weight loss", "Hemoptysis", "Weakness"],
        "treatment": ["Chemotherapy", "Immunotherapy", "Radiation therapy", "Surgery if early"],
        "findings": ["Large peripheral mass", "Necrotic center", "Mediastinal involvement"]
    },
    "Normal": {
        "severity": "Normal",
        "survival_rate": "Normal life expectancy",
        "symptoms": ["No significant symptoms", "Routine monitoring advised"],
        "treatment": ["Regular checkups", "Annual CT screening", "Healthy lifestyle"],
        "findings": ["No mass detected", "Normal lung parenchyma", "No pleural effusion"]
    },
    "Squamous Cell Carcinoma": {
        "severity": "High",
        "survival_rate": "20-25% (5-year)",
        "symptoms": ["Chronic cough", "Blood in sputum", "Wheezing", "Chest pain"],
        "treatment": ["Surgery", "Chemoradiation", "Immunotherapy", "Bronchoscopic treatment"],
        "findings": ["Central airway mass", "Possible airway obstruction", "Bronchoscopy recommended"]
    }
}

@app.get("/")
def home():
    return {"message": "AI CT Scan Analyzer Ready!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((64, 64))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = img_array[np.newaxis, np.newaxis, :, :]

    outputs = session.run(None, {"input": img_tensor})
    prediction = np.argmax(outputs[0])
    confidence = float(np.max(outputs[0]))

    diagnosis = CLASSES[prediction]
    data = MEDICAL_DATA[diagnosis]

    return {
        "diagnosis": diagnosis,
        "confidence": f"{abs(confidence)*10:.1f}%",
        "severity": data["severity"],
        "survival_rate": data["survival_rate"],
        "symptoms": data["symptoms"],
        "treatment": data["treatment"],
        "findings": data["findings"],
        "report_date": datetime.now().strftime("%d %B %Y, %H:%M")
    }