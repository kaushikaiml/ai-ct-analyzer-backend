import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_PATH = r"C:\Users\mohit\ai-ct-analyzer\data"

CLASSES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal", 
    "Squamous Cell Carcinoma"
]

# Same model
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

# Model load karo
model = CTModel()
model.load_state_dict(torch.load("ct_model.pth", weights_only=True))
model.eval()

# Dataset load karo
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Accuracy check karo
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print("===== ACCURACY REPORT =====")
print(f"Total Images: {total}")
print(f"Sahi Predictions: {correct}")
print(f"Galat Predictions: {total - correct}")
print(f"Accuracy: {accuracy:.1f}%")
print("===========================")

if accuracy > 70:
    print("Model ACCHA hai!")
elif accuracy > 50:
    print("Model THEEK hai — aur training chahiye!")
else:
    print("Model KAM hai — zyada training chahiye!")