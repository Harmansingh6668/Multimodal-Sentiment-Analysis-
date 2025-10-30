import os
import tempfile
import cv2
from collections import defaultdict
from image_analyzer import ImageSentiment, LABELS


class VideoSentiment:
    def __init__(self, frame_sample_rate=30, max_frames=12):
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames
        self.image_model = ImageSentiment()

    def predict_proba(self, video_path):
        """Extract frames, run image sentiment, and average distributions."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // self.max_frames)

        results = []
        idx = 0

        while cap.isOpened() and len(results) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # Save temporary frame
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    frame_path = tmp.name
                    cv2.imwrite(frame_path, frame)

                try:
                    dist = self.image_model.predict_proba(frame_path)
                    results.append(dist)
                except Exception as e:
                    print(f"⚠️ Skipping bad frame {idx}: {e}")
                finally:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            idx += 1

        cap.release()

        if not results:
            return {k: 0.0 for k in LABELS}

        # Average distributions
        avg = defaultdict(float)
        for dist in results:
            for k, v in dist.items():
                avg[k] += v
        for k in avg:
            avg[k] /= len(results)

        return dict(avg)
