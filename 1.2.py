import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152, ResNet152_Weights
import types
from utils import get_cifar10_dataloaders, plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = get_cifar10_dataloaders()

model = resnet152(weights=ResNet152_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)

def disabled_skip_forward(self, x):
    """A custom forward pass that completely ignores the identity mapping."""
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv3(out)
    out = self.bn3(out)
    
    out = self.relu(out)
    return out

for block in model.layer4:
    block.forward = types.MethodType(disabled_skip_forward, block)

print("Skip connections in Layer 4 disabled.")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
epochs = 3

print("Training without skip connections...")
for epoch in range(epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(trainloader)
    epoch_train_acc = 100 * correct_train / total_train
    
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(testloader)
    epoch_val_acc = 100 * correct_val / total_val
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_acc)
    val_accuracies.append(epoch_val_acc)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)