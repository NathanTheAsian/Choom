import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from choom.models.cnn import ChoomCNN

# ---------- Hyperparameters ----------
batch_size = 64
lr = 0.001
epochs = 10
num_classes = 3755  # adjust based on dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Data Preprocessing ----------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Replace with your dataset path
train_dataset = datasets.ImageFolder(root='data/processed/train', transform=transform)
val_dataset = datasets.ImageFolder(root='data/processed/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------- Model ----------
model = ChoomCNN(num_classes=num_classes).to(device)

# ---------- Loss & Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- Training Loop ----------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # ---------- Validation ----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ---------- Save Model ----------
torch.save(model.state_dict(), 'choom_cnn_baseline.pth')
print("Model saved as choom_cnn_baseline.pth")
