import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet152, ResNet152_Weights
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading CIFAR-100...")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

experiments = {
    "Pretrained_HeadOnly": {"pretrained": True, "freeze_backbone": True},
    "Pretrained_Full": {"pretrained": True, "freeze_backbone": False},
    "Random_Full": {"pretrained": False, "freeze_backbone": False},
}

results = {}
epochs = 2 # Keeping it to 2 epochs due to the massive compute required for full backbone training

def run_experiment(name, config):
    print(f"\n--- Running: {name} ---")
    weights = ResNet152_Weights.DEFAULT if config["pretrained"] else None
    model = resnet152(weights=weights)
    
    if config["freeze_backbone"]:
        for param in model.parameters():
            param.requires_grad = False
            
    model.fc = nn.Linear(model.fc.in_features, 100) # CIFAR-100 has 100 classes
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # If training full backbone, we need a smaller learning rate to avoid destroying weights
    lr = 0.001 if config["freeze_backbone"] else 0.0001
    optimizer = optim.Adam(model.parameters() if not config["freeze_backbone"] else model.fc.parameters(), lr=lr)
    
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        print(f"Epoch {epoch+1} | Val Acc: {acc:.2f}%")
        
    results[name] = val_accuracies

for name, config in experiments.items():
    run_experiment(name, config)

plt.figure(figsize=(8, 6))
for name, accs in results.items():
    plt.plot(range(1, epochs+1), accs, marker='o', label=name)

plt.title('Transfer Learning Strategies on CIFAR-100')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('transfer_learning.png', bbox_inches='tight', dpi=300)
print("\nSaved visualization as transfer_learning.png!")
