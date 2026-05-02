import torch
import torch.nn as nn

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

# Model load karo
model = CTScanCNN()
model.load_state_dict(torch.load("ct_model.pth", weights_only=True))
model.eval()

# ONNX mein convert karo
dummy_input = torch.randn(1, 1, 64, 64)
torch.onnx.export(
    model,
    dummy_input,
    "ct_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("Model convert ho gaya!")
print("File: ct_model.onnx")