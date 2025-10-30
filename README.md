# Multimodal Sentiment Analysis (Text + Video + Image) — VS Code Ready

This project performs **multimodal sentiment analysis** by combining:
- **Text** sentiment (Hugging Face SST-2 pipeline),
- **Image** sentiment (CLIP zero-shot),
- **Video** sentiment (sample frames → CLIP zero-shot per frame → average).

It uses **weighted late fusion** to produce **Negative / Neutral / Positive**.

---

## 📁 Project Structure
```
multimodal_sentiment/
│── data/
│   ├── sample_text.txt
│   ├── sample_image.jpg
│   └── sample_video.mp4
│── src/
│   ├── text_analyzer.py
│   ├── image_analyzer.py
│   ├── video_analyzer.py
│   ├── fusion.py
│   └── inference.py
│── .vscode/
│   └── launch.json
│── requirements.txt
│── README.md
│── download_models.py
│── run.bat
│── run.sh
```

---

## ✅ Step-by-step Setup (VS Code)

### 0) Open the folder in VS Code
**File → Open Folder** → select this project folder.

### 1) Create & Select Virtual Environment
**Windows:**
```bat
python -m venv venv
venv\Scripts\activate
```
**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```
Then: **Ctrl+Shift+P → Python: Select Interpreter** → choose the one from `venv`.

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Recommended) Pre-download models
```bash
python download_models.py
```

### 4) Run examples
**Text + Video:**
```bash
python src/inference.py --text-file data/sample_text.txt --video data/sample_video.mp4
```
**Text + Image:**
```bash
python src/inference.py --text-file data/sample_text.txt --image data/sample_image.jpg
```
**All three:**
```bash
python src/inference.py --text "This is great!" --image data/sample_image.jpg --video data/sample_video.mp4
```

### 5) Modality weights
You can tune weights (they auto-normalize if a modality is missing):
```bash
python src/inference.py --text "Not bad" --video data/sample_video.mp4 --w-text 0.5 --w-image 0.2 --w-video 0.3
```

---

## 🧠 How it works (brief)
- **Text** → DistilBERT SST-2 pipeline → 2-class mapped to 3-class with a neutral band.
- **Image** → CLIP zero-shot with prompts for {negative, neutral, positive}.
- **Video** → sample frames (every N frames, up to M total), CLIP per frame, average to video distribution.
- **Fusion** → Weighted average across available modalities.

---

## ❓ Troubleshooting
If you encounter model download issues, clear cache:
- Windows: `C:\Users\<you>\.cache\huggingface\` and `C:\Users\<you>\.cache\torch\`

---

Enjoy!
