import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import load_sample_image, extract_cls_attention_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/vit-base-patch16-224'

print(f"Loading {model_name}...")
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True).to(device)
model.eval()

image = load_sample_image()
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pred_idx = outputs.logits.argmax(-1).item()
pred_label = model.config.id2label[pred_idx]
print(f"Top-1 Prediction: {pred_label}")

# Extract spatial attention map via utils helper
att_map_14x14 = extract_cls_attention_map(outputs.attentions, layer_idx=-1)

# Upsample to image dimensions
att_resized = Image.fromarray(np.uint8(255 * att_map_14x14), mode='L').resize((224, 224), Image.Resampling.BILINEAR)
att_overlay = np.array(att_resized) / 255.0

orig_224 = image.resize((224, 224))

plt.figure(figsize=(12, 4.5))
plt.subplot(1, 3, 1)
plt.imshow(orig_224)
plt.title(f"Input Image\nPred: {pred_label}", fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(att_map_14x14, cmap='hot')
plt.title("14x14 [CLS] Attention Map", fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(orig_224)
plt.imshow(att_overlay, cmap='Reds', alpha=0.55)
plt.title("Attention Heatmap Overlay", fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('vit_attention.png', bbox_inches='tight', dpi=300)
