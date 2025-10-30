"""Pre-download all models so the first inference run is smooth."""
from transformers import pipeline, CLIPModel, CLIPProcessor

def main():
    _ = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
    _ = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _ = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("All models downloaded and cached.")

if __name__ == "__main__":
    main()
