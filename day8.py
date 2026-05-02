import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

CLASSES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

class CTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.network(x)

model = CTModel()
model.load_state_dict(torch.load("ct_model.pth", weights_only=True))
model.eval()
print("Model load ho gaya!")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
img_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    prediction = torch.argmax(output).item()
    confidence = torch.softmax(output, dim=1)[0][prediction].item()

print("\n===== RESULT =====")
print(f"Diagnosis: {CLASSES[prediction]}")
print(f"Confidence: {confidence*100:.1f}%")
print("==================")