import torch
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'google/vit-base-patch16-224'
print(f"Loading {model_name}...")
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name).to(device)
model.eval()

url = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG"
urllib.request.urlretrieve(url, "sample_image.jpg")
original_image = Image.open("sample_image.jpg").convert("RGB").resize((224, 224))

def apply_patch_mask(image, strategy="random", mask_ratio=0.5, patch_size=16):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w
    num_mask = int(total_patches * mask_ratio)

    mask = np.ones((num_patches_h, num_patches_w), dtype=bool)

    if strategy == "random":
        flat_mask = np.ones(total_patches, dtype=bool)
        mask_indices = np.random.choice(total_patches, num_mask, replace=False)
        flat_mask[mask_indices] = False
        mask = flat_mask.reshape((num_patches_h, num_patches_w))
        
    elif strategy == "structured":
        start = 2
        end = 12
        mask[start:end, start:end] = False

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if not mask[i, j]:
                img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 0
                
    return Image.fromarray(img_array)

img_random = apply_patch_mask(original_image, strategy="random", mask_ratio=0.5)
img_structured = apply_patch_mask(original_image, strategy="structured", mask_ratio=0.5)

images = [original_image, img_random, img_structured]
titles = ["Original (0% Masked)", "Random Masking (~50%)", "Structured Masking (Center ~50%)"]

plt.figure(figsize=(15, 5))

for idx, (img, title) in enumerate(zip(images, titles)):
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_prob, top_catid = torch.max(probs, dim=0)
    predicted_label = model.config.id2label[top_catid.item()]
    
    plt.subplot(1, 3, idx + 1)
    plt.imshow(img)
    short_label = predicted_label.split(',')[0] 
    plt.title(f"{title}\nPred: {short_label} ({top_prob.item()*100:.1f}%)", fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('patch_masking.png', bbox_inches='tight', dpi=300)
