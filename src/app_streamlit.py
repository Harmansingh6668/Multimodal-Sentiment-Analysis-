import os
import sys
import tempfile
import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Ensure local imports work
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from text_analyzer import TextSentiment, LABELS
from image_analyzer import ImageSentiment
from video_analyzer import VideoSentiment
from fusion import fuse_many, argmax_label


# ---------- Helpers ----------
def draw_architecture_diagram():
    """Draw simple architecture diagram with multiline text support."""
    W, H = 1200, 540
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def box(x0, y0, x1, y1, text):
        d.rounded_rectangle((x0, y0, x1, y1), radius=24, outline="black", width=3)
        lines = text.split("\n")
        line_height = 22
        total_height = len(lines) * line_height
        y_start = (y0 + y1) / 2 - total_height / 2
        for i, line in enumerate(lines):
            # measure each line separately (textlength supports single-line)
            tw = d.textlength(line)
            cx = (x0 + x1) / 2
            cy = y_start + i * line_height
            d.text((cx - tw / 2, cy), line, fill="black")

    def arrow(x0, y0, x1, y1):
        d.line((x0, y0, x1, y1), fill="black", width=3)
        ah = 10
        d.polygon([(x1, y1), (x1 - ah, y1 - ah), (x1 - ah, y1 + ah)], fill="black")

    # Inputs
    box(40, 60, 260, 140, "Text Input")
    box(40, 220, 260, 300, "Image Input")
    box(40, 380, 260, 460, "Video Input")

    # Encoders
    box(330, 60, 600, 140, "Text Encoder\n(DistilBERT SST-2)")
    box(330, 220, 600, 300, "Image Encoder\n(CLIP)")
    box(330, 380, 600, 460, "Video Encoder\n(CLIP on Frames)")

    # Fusion + Classifier
    box(690, 180, 930, 260, "Feature Fusion\n(Weighted)")
    box(1030, 180, 1180, 260, "Sentiment\nPrediction")

    # Arrows
    arrow(260, 100, 330, 100)
    arrow(260, 260, 330, 260)
    arrow(260, 420, 330, 420)

    arrow(600, 100, 690, 220)
    arrow(600, 260, 690, 220)
    arrow(600, 420, 690, 220)

    arrow(930, 220, 1030, 220)

    d.text((40, 12), "Multimodal Sentiment Analysis — Text + Image + Video", fill="black")
    return img


def bar_chart(title, dist):
    fig, ax = plt.subplots()
    labels = list(dist.keys())
    values = [dist[k] for k in labels]
    ax.bar(labels, values)          # no custom colors
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig)


def save_uploaded_file(uploaded, suffix):
    if uploaded is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


def remove_file_safe(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Multimodal Sentiment", layout="wide")
st.title("🎭 Multimodal Sentiment Analysis — Text + Image + Video")

tabs = st.tabs(["📐 Design", "🧪 Try it"])

# --- Design Tab ---
with tabs[0]:
    st.subheader("System Architecture")
    img = draw_architecture_diagram()
    st.image(img, caption="Text/Image/Video → Encoders → Fusion → Prediction", use_column_width=True)

    st.markdown(
        """
**Flow:**  
1. **Text** → DistilBERT SST-2 → mapped to 3-class with neutral band  
2. **Image** → CLIP zero-shot over {negative, neutral, positive}  
3. **Video** → sample frames → CLIP per frame → averaged  
4. **Fusion** → weighted average  
5. **Prediction** → argmax over fused distribution
        """
    )

# --- Try it Tab ---
with tabs[1]:
    st.subheader("Upload Your Inputs")

    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_area("Text (optional)")
        text_file = st.file_uploader("Or upload a text file", type=["txt"])
    with col2:
        image_file = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])
        video_file = st.file_uploader("Video (optional)", type=["mp4", "avi", "mov", "mkv"])

    # Weights
    st.markdown("**Fusion Weights**")
    wt, wi, wv = st.columns(3)
    with wt:
        w_text = st.slider("Text", 0.0, 1.0, 0.6, 0.05)
    with wi:
        w_image = st.slider("Image", 0.0, 1.0, 0.2, 0.05)
    with wv:
        w_video = st.slider("Video", 0.0, 1.0, 0.2, 0.05)

    if st.button("Run Sentiment Analysis"):
        # Prepare inputs
        text_str = None
        if text_input and text_input.strip():
            text_str = text_input.strip()
        elif text_file is not None:
            text_str = text_file.read().decode("utf-8", errors="ignore").strip()

        image_path = save_uploaded_file(image_file, ".jpg") if image_file else None
        video_path = save_uploaded_file(video_file, ".mp4") if video_file else None

        # Initialize analyzers
        txt = TextSentiment()
        img_model = ImageSentiment()
        vid_model = VideoSentiment(frame_sample_rate=30, max_frames=12)

        # Predict
        text_dist = {k: 0.0 for k in LABELS}
        image_dist = {k: 0.0 for k in LABELS}
        video_dist = {k: 0.0 for k in LABELS}

        if text_str:
            try:
                text_dist = txt.predict_proba(text_str)
            except Exception as e:
                st.error(f"⚠️ Text sentiment analysis failed: {e}")
                text_dist = {k: 0.0 for k in LABELS}

        if image_path:
            try:
                image_dist = img_model.predict_proba(image_path)
            except Exception as e:
                st.error(f"⚠️ Image sentiment analysis failed: {e}")
                image_dist = {k: 0.0 for k in LABELS}

        # SAFE video call: try/except and check None
        try:
            if video_path is not None:
                video_dist = vid_model.predict_proba(video_path)
            else:
                video_dist = {k: 0.0 for k in LABELS}
        except Exception as e:
            st.error(f"⚠️ Video sentiment analysis failed: {e}")
            video_dist = {k: 0.0 for k in LABELS}

        fused = fuse_many([text_dist, image_dist, video_dist], [w_text, w_image, w_video])
        pred = argmax_label(fused)

        # Show results
        st.success(f"**Prediction:** {pred.capitalize()}")

        # fixed layout: create columns then call bar_chart inside with-blocks
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bar_chart("Text", text_dist)
        with c2:
            bar_chart("Image", image_dist)
        with c3:
            bar_chart("Video", video_dist)
        with c4:
            bar_chart("Fused", fused)

        st.code(json.dumps({
            "inputs": {"text": text_str, "image": bool(image_path), "video": bool(video_path)},
            "probabilities": {"text": text_dist, "image": image_dist, "video": video_dist, "fused": fused},
            "prediction": pred
        }, indent=2), language="json")

        # cleanup temp files
        remove_file_safe(image_path)
        remove_file_safe(video_path)
