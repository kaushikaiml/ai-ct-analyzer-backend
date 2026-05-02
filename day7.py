import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_PATH = r"C:\Users\mohit\ai-ct-analyzer\data"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model banao
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
print("Training shuru!")
for epoch in range(5):
    total_loss = 0
    for images, labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5 - Loss: {total_loss:.2f}")

print("Training complete!")
torch.save(model.state_dict(), "ct_model.pth")
print("Model save ho gaya!")