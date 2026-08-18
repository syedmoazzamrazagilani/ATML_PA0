import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request

# --- CIFAR / CNN Helpers (Task 1) ---
def get_cifar10_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, filename="metrics.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved metric plot to {filename}")

# --- ViT Helpers (Task 2) ---
def load_sample_image(url="https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG", save_path="sample.jpg"):
    urllib.request.urlretrieve(url, save_path)
    return Image.open(save_path).convert("RGB")

def extract_cls_attention_map(attentions, layer_idx=-1, grid_size=(14, 14)):
    """Extracts, averages heads, and spatializes the [CLS] attention from a specified layer."""
    # attentions shape: tuple of (batch_size, num_heads, seq_len, seq_len)
    layer_att = attentions[layer_idx]
    # Average across all attention heads: (batch_size, seq_len, seq_len)
    mean_att = layer_att.mean(dim=1)
    # [CLS] token is at index 0; patch tokens are indices 1..196
    cls_to_patches = mean_att[0, 0, 1:].detach().cpu().numpy()
    # Normalize between 0 and 1
    norm_att = (cls_to_patches - cls_to_patches.min()) / (cls_to_patches.max() - cls_to_patches.min() + 1e-8)
    return norm_att.reshape(grid_size)

def apply_patch_mask(image, strategy="random", mask_ratio=0.5, patch_size=16):
    """Applies random or structured (central block) patch masking to a PIL Image."""
    img_array = np.array(image.resize((224, 224)))
    h, w, _ = img_array.shape
    num_patches_h, num_patches_w = h // patch_size, w // patch_size
    total_patches = num_patches_h * num_patches_w
    
    mask = np.ones((num_patches_h, num_patches_w), dtype=bool)
    if strategy == "random":
        num_mask = int(total_patches * mask_ratio)
        flat_mask = np.ones(total_patches, dtype=bool)
        mask_indices = np.random.choice(total_patches, num_mask, replace=False)
        flat_mask[mask_indices] = False
        mask = flat_mask.reshape((num_patches_h, num_patches_w))
    elif strategy == "structured":
        # Block center mask (10x10 patches = ~51% area)
        mask[2:12, 2:12] = False

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if not mask[i, j]:
                img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 0
                
    return Image.fromarray(img_array)
