import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import get_cifar10_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, testloader = get_cifar10_dataloaders(batch_size=128)

print("Loading pre-trained ResNet-152...")
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)
model = model.to(device)
model.eval()

features = {'early': [], 'middle': [], 'late': []}

def hook_early(module, input, output):
    features['early'].append(torch.mean(output, dim=[2, 3]).detach().cpu())

def hook_middle(module, input, output):
    features['middle'].append(torch.mean(output, dim=[2, 3]).detach().cpu())

def hook_late(module, input, output):
    features['late'].append(torch.mean(output, dim=[2, 3]).detach().cpu())

model.layer1.register_forward_hook(hook_early)
model.layer3.register_forward_hook(hook_middle)
model.layer4.register_forward_hook(hook_late)

all_labels = []
max_samples = 1000
samples_collected = 0

print("Extracting layer representations...")
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        _ = model(inputs)
        all_labels.append(labels.numpy())
        samples_collected += inputs.size(0)
        if samples_collected >= max_samples:
            break

all_labels = np.concatenate(all_labels)[:max_samples]
for key in features:
    features[key] = torch.cat(features[key], dim=0)[:max_samples].numpy()

cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
layers = [('early', 'Early Layer (Layer 1)'), 
          ('middle', 'Middle Layer (Layer 3)'), 
          ('late', 'Late Layer (Layer 4)')]

plt.figure(figsize=(18, 5))

for i, (layer_key, layer_name) in enumerate(layers, 1):
    print(f"Computing t-SNE for {layer_name}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(features[layer_key])
    
    plt.subplot(1, 3, i)
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=15)
    plt.title(layer_name, fontsize=12, fontweight='bold')
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

plt.subplots_adjust(right=0.88)
cbar_ax = plt.gcf().add_axes([0.90, 0.15, 0.015, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax, ticks=range(10))
cbar.ax.set_yticklabels(cifar10_classes)

plt.savefig('tsne_features.png', bbox_inches='tight', dpi=300)
