import torch
import torchvision
import torchvision.transforms as transforms
from transformers import ViTModel, ViTImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/vit-base-patch16-224'

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(2000)), batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(500)), batch_size=64, shuffle=False)

def extract_tokens(dataloader):
    cls_list, mean_list, labels_list = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            hidden = model(inputs).last_hidden_state
            cls_list.append(hidden[:, 0, :].cpu().numpy())
            mean_list.append(hidden[:, 1:, :].mean(dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(cls_list), np.vstack(mean_list), np.concatenate(labels_list)

print("Extracting representations...")
X_train_cls, X_train_mean, y_train = extract_tokens(trainloader)
X_test_cls, X_test_mean, y_test = extract_tokens(testloader)

clf_cls = LogisticRegression(max_iter=1000).fit(X_train_cls, y_train)
acc_cls = accuracy_score(y_test, clf_cls.predict(X_test_cls)) * 100

clf_mean = LogisticRegression(max_iter=1000).fit(X_train_mean, y_train)
acc_mean = accuracy_score(y_test, clf_mean.predict(X_test_mean)) * 100

print(f"CLS Accuracy: {acc_cls:.2f}% | Mean-Pooled Accuracy: {acc_mean:.2f}%")

plt.figure(figsize=(5.5, 4.5))
bars = plt.bar(['[CLS] Token', 'Mean-Pooled Patches'], [acc_cls, acc_mean], color=['#1f77b4', '#ff7f0e'], width=0.55)
plt.ylabel('Classification Accuracy (%)', fontweight='bold')
plt.title('Linear Probing on ViT Representations', fontweight='bold')
plt.ylim(0, 110)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('pooling_comparison.png', bbox_inches='tight', dpi=300)
