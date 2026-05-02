import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ================================
# STEP 1: Image ko AI ke liye ready karo
# ================================
image = Image.open(r"C:\Users\mohit\ai-ct-analyzer\data\ct_scan.png")
image = image.convert("L").resize((64, 64))  # Chhota karo
img_array = np.array(image).astype(np.float32) / 255.0  # 0-1 ke beech karo
img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # AI format

print("Image tensor shape:", img_tensor.shape)

# ================================
# STEP 2: Neural Network banao
# ================================
class CTScanModel(nn.Module):
    def __init__(self):
        super(CTScanModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 outputs: Cancer ya Normal
        )

    def forward(self, x):
        return self.network(x)

# ================================
# STEP 3: Model banao aur chalao
# ================================
model = CTScanModel()
print("\nModel ready hai!")
print(model)

# ================================
# STEP 4: Prediction karo
# ================================
output = model(img_tensor)
prediction = torch.argmax(output).item()

if prediction == 0:
    print("\nResult: NORMAL")
else:
    print("\nResult: CANCER DETECTED")

print("\nDay 5 Complete!")