from typing import Dict, Optional
import torch
from transformers import pipeline

LABELS = ["negative", "neutral", "positive"]

class TextSentiment:
    """Hugging Face sentiment pipeline (SST-2 → 3-class with neutral band)."""
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == "cuda" else -1,
            return_all_scores=True
        )

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {k: 0.0 for k in LABELS}

        outputs = self.pipe(text)[0]
        scores = {o["label"].lower(): float(o["score"]) for o in outputs}

        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)

        conf = max(pos, neg)
        margin = abs(pos - neg)

        if conf < 0.6 or margin < 0.2:
            neutral = 1.0 - margin
            s = pos + neg + neutral
            return {
                "negative": neg / s,
                "neutral": neutral / s,
                "positive": pos / s,
            }
        else:
            neutral = 1.0 - (pos + neg)
            if neutral < 0.0: neutral = 0.0
            s = pos + neg + neutral
            return {
                "negative": neg / s,
                "neutral": neutral / s,
                "positive": pos / s,
            }
