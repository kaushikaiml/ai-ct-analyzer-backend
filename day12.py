import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_PATH = r"C:\Users\mohit\ai-ct-analyzer\data"

# Zyada powerful transform — data augmentation
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset load karo
dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform_train)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Zyada powerful model
class CTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x):
        return self.network(x)

model = CTModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

# 20 epochs train karo
print("Training shuru!")
for epoch in range(20):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Accuracy check karo
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.2f} - Accuracy: {accuracy:.1f}%")

# Best model save karo
torch.save(model.state_dict(), "ct_model.pth")
print("\nModel save ho gaya!")
print(f"Final Accuracy: {accuracy:.1f}%")