import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset ka path
TRAIN_PATH = r"C:\Users\mohit\ai-ct-analyzer\data"

# Image ko AI ke liye ready karo
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset load karo
dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Categories:", dataset.classes)
print("Total images:", len(dataset))
print("Dataset ready hai!")
print("Day 6 Complete!")