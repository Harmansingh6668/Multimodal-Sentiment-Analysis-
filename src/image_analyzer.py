from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

LABELS = ["negative", "neutral", "positive"]

class ImageSentiment:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict_proba(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"⚠️ Skipping invalid image {image_path}: {e}")
            return {k: 0.0 for k in LABELS}

        # Prepare text labels
        texts = [f"A {label} expression" for label in LABELS]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = F.softmax(logits_per_image, dim=1).cpu().numpy()[0]

        return {label: float(prob) for label, prob in zip(LABELS, probs)}
