import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import pandas as pd
import numpy as np
from PIL import Image

# 1. Dataset Setup
os.makedirs("models", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)

# ImageNet transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Export 5 Real PNG Samples from Test Split
raw_test = datasets.FashionMNIST(root="data", train=False, download=True)
class_names = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

exported_counts = {}
for idx, (img, label_idx) in enumerate(raw_test):
    cname = class_names[label_idx]
    if cname not in exported_counts and len(exported_counts) < 5:
        exported_counts[cname] = True
        img_path = f"data/sample_images/0{len(exported_counts)}_{cname}.png"
        img.save(img_path)
        print(f"Exported sample image: {img_path}")

# Split Train into Train (55,000) and Val (5,000)
indices = list(range(len(train_full)))
val_indices = indices[:5000]
train_indices = indices[5000:]

train_ds = Subset(train_full, train_indices)
val_ds = Subset(train_full, val_indices)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Pretrained Backbone Setup (ResNet-18)
device = torch.device("cpu") # M4 CPU handles this lightweight fine-tuning quickly
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("\n--- TRAINING FEATURE EXTRACTION HEAD ---")
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    # Validation Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_ds):.4f} - Val Acc: {val_acc:.4f}")

# Final Test Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nFinal Test Set Accuracy: {test_acc:.4f}")

# Save PyTorch Weights
torch.save(model.state_dict(), "models/product_classifier.pt")
print("Saved classifier model to models/product_classifier.pt")
