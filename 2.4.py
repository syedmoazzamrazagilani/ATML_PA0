import torch
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import load_sample_image, apply_patch_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/vit-base-patch16-224'

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

orig_img = load_sample_image().resize((224, 224))
img_random = apply_patch_mask(orig_img, strategy="random", mask_ratio=0.5)
img_structured = apply_patch_mask(orig_img, strategy="structured")

images = [orig_img, img_random, img_structured]
titles = ["Original (0% Mask)", "Random Mask (50%)", "Structured Center Mask (50%)"]

plt.figure(figsize=(14, 4.5))
for idx, (img, title) in enumerate(zip(images, titles)):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_p, top_id = torch.max(probs, dim=0)
    pred_label = model.config.id2label[top_id.item()].split(',')[0]

    plt.subplot(1, 3, idx + 1)
    plt.imshow(img)
    plt.title(f"{title}\nPred: {pred_label} ({top_p.item()*100:.1f}%)", fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('patch_masking.png', bbox_inches='tight', dpi=300)
