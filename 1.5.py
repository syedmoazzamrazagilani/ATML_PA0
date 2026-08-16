import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights, resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from utils import get_cifar10_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
_, testloader = get_cifar10_dataloaders(batch_size=128)
cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Load Pretrained Models
print("Loading pre-trained ResNet-152 and ResNet-18...")
model_152 = resnet152(weights=ResNet152_Weights.DEFAULT).to(device).eval()
model_18 = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()

# Remove classification heads to get raw feature embeddings from the final layer
model_152.fc = nn.Identity()
model_18.fc = nn.Identity()

# 3. Extract Features
max_samples = 1500
features_152, features_18, all_labels = [], [], []
samples_collected = 0

print(f"Extracting features for {max_samples} samples...")
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        features_152.append(model_152(inputs).cpu().numpy())
        features_18.append(model_18(inputs).cpu().numpy())
        all_labels.append(labels.numpy())
        
        samples_collected += inputs.size(0)
        if samples_collected >= max_samples:
            break

features_152 = np.concatenate(features_152)[:max_samples]
features_18 = np.concatenate(features_18)[:max_samples]
all_labels = np.concatenate(all_labels)[:max_samples]

plt.figure(figsize=(20, 15))

# --- Part 5(a): Compare t-SNE vs. UMAP for ResNet-152 ---
print("Computing t-SNE for ResNet-152...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_152_tsne = tsne.fit_transform(features_152)

print("Computing UMAP for ResNet-152...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb_152_umap = reducer.fit_transform(features_152)

plt.subplot(2, 2, 1)
scatter = plt.scatter(emb_152_tsne[:, 0], emb_152_tsne[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=15)
plt.title("ResNet-152 Features: t-SNE", fontsize=14, fontweight='bold')

plt.subplot(2, 2, 2)
plt.scatter(emb_152_umap[:, 0], emb_152_umap[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=15)
plt.title("ResNet-152 Features: UMAP", fontsize=14, fontweight='bold')

# --- Part 5(b): Analyze feature similarities (Confusion Analysis) ---
print("Computing Class Feature Similarities...")
class_centroids = []
for i in range(10):
    class_features = features_152[all_labels == i]
    class_centroids.append(np.mean(class_features, axis=0))

# Compute cosine similarity between the average feature vector of each class
similarity_matrix = cosine_similarity(class_centroids)

plt.subplot(2, 2, 3)
sns.heatmap(similarity_matrix, xticklabels=cifar10_classes, yticklabels=cifar10_classes, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("ResNet-152 Feature Similarity Between Classes", fontsize=14, fontweight='bold')

# --- Part 5(c): Compare Feature Quality with ResNet-18 ---
print("Computing UMAP for ResNet-18...")
emb_18_umap = reducer.fit_transform(features_18)

plt.subplot(2, 2, 4)
plt.scatter(emb_18_umap[:, 0], emb_18_umap[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=15)
plt.title("ResNet-18 Features: UMAP", fontsize=14, fontweight='bold')

# Add a shared colorbar for the scatter plots
plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9, hspace=0.3)
cbar_ax = plt.gcf().add_axes([0.88, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax, ticks=range(10))
cbar.ax.set_yticklabels(cifar10_classes)

plt.savefig('optional_experiments.png', bbox_inches='tight', dpi=300)
