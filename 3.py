import torch
import clip
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading STL-10 dataset...")
testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=preprocess)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

prompts = {
    "Plain Label": "{}",
    "Standard Prompt": "a photo of a {}",
    "Descriptive Prompt": "a centered, high quality photo of a {}, a type of animal or vehicle"
}

def get_text_features(prompt_template):
    texts = [prompt_template.format(c) for c in stl10_classes]
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

print("\nExtracting image features for STL-10 test set...")
all_image_features, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(testloader):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features.cpu())
        all_labels.append(labels.numpy())

image_features_np = torch.cat(all_image_features).numpy()
labels_np = np.concatenate(all_labels)

accuracies = {}
text_features_dict = {}

print("\nEvaluating Prompt Strategies...")
for name, template in prompts.items():
    tf = get_text_features(template).cpu().numpy()
    text_features_dict[name] = tf
    
    similarity = image_features_np @ tf.T
    preds = np.argmax(similarity, axis=1)
    acc = np.mean(preds == labels_np) * 100
    accuracies[name] = acc
    print(f"Zero-Shot Accuracy ({name}): {acc:.2f}%")

plt.figure(figsize=(8, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("CLIP Zero-Shot Accuracy on STL-10\nEffect of Prompt Engineering", fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.savefig('clip_zeroshot.png', bbox_inches='tight', dpi=300)
print("Saved clip_zeroshot.png")

print("\nAnalyzing Modality Gap...")
T_features = text_features_dict["Standard Prompt"]

np.random.seed(42)
subset_indices = np.random.choice(len(image_features_np), 250, replace=False)
I_subset = image_features_np[subset_indices]
L_subset = labels_np[subset_indices]

X_combined_pre = np.vstack([I_subset, T_features])
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_pre = reducer.fit_transform(X_combined_pre)

I_umap_pre = umap_pre[:len(I_subset)]
T_umap_pre = umap_pre[len(I_subset):]


print("Computing Procrustes alignment (SVD)...")
Y_paired = T_features[labels_np] 
X_paired = image_features_np     

R, scale = orthogonal_procrustes(X_paired, Y_paired)

I_aligned = X_paired @ R

similarity_aligned = I_aligned @ T_features.T
preds_aligned = np.argmax(similarity_aligned, axis=1)
acc_aligned = np.mean(preds_aligned == labels_np) * 100
print(f"Zero-Shot Accuracy (Post-Procrustes): {acc_aligned:.2f}%")

I_subset_aligned = I_aligned[subset_indices]
X_combined_post = np.vstack([I_subset_aligned, T_features])
umap_post = reducer.fit_transform(X_combined_post)

I_umap_post = umap_post[:len(I_subset_aligned)]
T_umap_post = umap_post[len(I_subset_aligned):]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pre-Alignment
scatter = axes[0].scatter(I_umap_pre[:, 0], I_umap_pre[:, 1], c=L_subset, cmap='tab10', alpha=0.6, s=20)
axes[0].scatter(T_umap_pre[:, 0], T_umap_pre[:, 1], c='red', marker='X', s=150, edgecolors='black', linewidth=1.5, label='Text Prompts')
axes[0].set_title("Pre-Alignment Modality Gap (UMAP)", fontweight='bold')
axes[0].legend()

# Post-Alignment
axes[1].scatter(I_umap_post[:, 0], I_umap_post[:, 1], c=L_subset, cmap='tab10', alpha=0.6, s=20)
axes[1].scatter(T_umap_post[:, 0], T_umap_post[:, 1], c='red', marker='X', s=150, edgecolors='black', linewidth=1.5, label='Text Prompts')
axes[1].set_title(f"Post-Procrustes Alignment (UMAP)\nAligned Accuracy: {acc_aligned:.2f}%", fontweight='bold')
axes[1].legend()

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(scatter, cax=cbar_ax, ticks=range(10))
cbar.ax.set_yticklabels(stl10_classes)

plt.subplots_adjust(right=0.90)
plt.savefig('clip_modality.png', bbox_inches='tight', dpi=300)
