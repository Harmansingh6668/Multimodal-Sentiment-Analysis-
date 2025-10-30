import torch
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

# Sentiment labels
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# Emotion labels (Ekman + Neutral)
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

class UnifiedAnalyzer:
    def __init__(self, device=-1):
        # ✅ Text
        self.text_sentiment = pipeline("sentiment-analysis",
                                       model="distilbert-base-uncased-finetuned-sst-2-english",
                                       device=device)
        self.text_emotion = pipeline("text-classification",
                                     model="j-hartmann/emotion-english-distilroberta-base",
                                     top_k=None,
                                     device=device)

        # ✅ Image/Video
        self.image_emotion = pipeline("image-classification",
                                      model="dima806/facial_emotions_image_detection",
                                      top_k=None,
                                      device=device)

    # -------------------- TEXT --------------------
    def analyze_text(self, text: str):
        out_sent = self.text_sentiment(text)[0]
        sent_dist = {k: 0.0 for k in SENTIMENT_LABELS}
        if out_sent["label"].lower() == "positive":
            sent_dist["positive"] = out_sent["score"]
            sent_dist["negative"] = 1 - out_sent["score"]
        else:
            sent_dist["negative"] = out_sent["score"]
            sent_dist["positive"] = 1 - out_sent["score"]

        # Add neutral band
        sent_dist["neutral"] = max(0.0, 1 - max(sent_dist.values()))

        # Emotions
        emo_preds = self.text_emotion(text, return_all_scores=True)[0]
        emo_dist = {k: 0.0 for k in EMOTION_LABELS}
        for p in emo_preds:
            lbl = p["label"].lower()
            if lbl in emo_dist:
                emo_dist[lbl] = float(p["score"])

        return {"sentiment": sent_dist, "emotion": emo_dist}

    # -------------------- IMAGE --------------------
    def analyze_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        preds = self.image_emotion(image, return_all_scores=True)[0]
        emo_dist = {k: 0.0 for k in EMOTION_LABELS}
        for p in preds:
            lbl = p["label"].lower()
            if lbl in emo_dist:
                emo_dist[lbl] = float(p["score"])

        # Simple sentiment mapping from emotions
        sent_dist = {
            "negative": emo_dist["anger"] + emo_dist["disgust"] + emo_dist["fear"] + emo_dist["sadness"],
            "neutral": emo_dist["neutral"],
            "positive": emo_dist["joy"] + emo_dist["surprise"]
        }
        total = sum(sent_dist.values())
        if total > 0:
            sent_dist = {k: v/total for k, v in sent_dist.items()}

        return {"sentiment": sent_dist, "emotion": emo_dist}

    # -------------------- VIDEO --------------------
    def analyze_video(self, video_path: str, frame_sample_rate=30, max_frames=12):
        cap = cv2.VideoCapture(video_path)
        frames = []
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_sample_rate == 0 and len(frames) < max_frames:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()

        # Aggregate predictions
        emo_dist = {k: 0.0 for k in EMOTION_LABELS}
        for f in frames:
            img = Image.fromarray(f)
            preds = self.image_emotion(img, return_all_scores=True)[0]
            for p in preds:
                lbl = p["label"].lower()
                if lbl in emo_dist:
                    emo_dist[lbl] += float(p["score"])
        if frames:
            emo_dist = {k: v/len(frames) for k, v in emo_dist.items()}

        # Map emotions → sentiment
        sent_dist = {
            "negative": emo_dist["anger"] + emo_dist["disgust"] + emo_dist["fear"] + emo_dist["sadness"],
            "neutral": emo_dist["neutral"],
            "positive": emo_dist["joy"] + emo_dist["surprise"]
        }
        total = sum(sent_dist.values())
        if total > 0:
            sent_dist = {k: v/total for k, v in sent_dist.items()}

        return {"sentiment": sent_dist, "emotion": emo_dist}
