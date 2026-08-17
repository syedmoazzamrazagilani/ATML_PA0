import torch
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/vit-base-patch16-224'

print(f"Loading {model_name}...")
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True).to(device)
model.eval()

url = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG"
urllib.request.urlretrieve(url, "sample_image.jpg")
image = Image.open("sample_image.jpg").convert("RGB")

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
print(f"\nTop-1 Prediction: {predicted_label}")

final_layer_attention = outputs.attentions[-1]

mean_attention = final_layer_attention.mean(dim=1)

cls_attention = mean_attention[0, 0, 1:]

attention_map = cls_attention.reshape(14, 14).cpu().numpy()

attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

attention_img = Image.fromarray(np.uint8(255 * attention_map), mode='L')
attention_img = attention_img.resize((224, 224), resample=Image.Resampling.BILINEAR)
attention_resized = np.array(attention_img) / 255.0

orig_img_resized = image.resize((224, 224))

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(orig_img_resized)
plt.title(f"Original Image\nPred: {predicted_label}")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(attention_map, cmap='hot')
plt.title("14x14 Attention Map")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(orig_img_resized)
plt.imshow(attention_resized, cmap='Reds', alpha=0.6)
plt.title("Attention Overlay")
plt.axis('off')

plt.tight_layout()
plt.savefig('vit_attention.png', bbox_inches='tight', dpi=300)
