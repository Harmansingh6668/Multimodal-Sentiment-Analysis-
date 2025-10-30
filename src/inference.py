import argparse, json, os
import torch
from text_analyzer import TextSentiment, LABELS as LABELS
from image_analyzer import ImageSentiment
from video_analyzer import VideoSentiment
from fusion import fuse_many, argmax_label

def main():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Inference (text + image + video)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--text", type=str, help="Raw input text")
    parser.add_argument("--text-file", type=str, help="Path to a text file", default=None)
    parser.add_argument("--image", type=str, help="Path to an image file (jpg/png)", default=None)
    parser.add_argument("--video", type=str, help="Path to a video file (mp4/avi)", default=None)
    parser.add_argument("--w-text", type=float, default=0.6, help="Weight for text modality (0..1)")
    parser.add_argument("--w-image", type=float, default=0.2, help="Weight for image modality (0..1)")
    parser.add_argument("--w-video", type=float, default=0.2, help="Weight for video modality (0..1)")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (auto if omitted)")
    parser.add_argument("--video-sample-rate", type=int, default=30, help="Sample 1 frame every N frames") 
    parser.add_argument("--video-max-frames", type=int, default=12, help="Max frames to analyze per video")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare text
    text_input = None
    if args.text is not None:
        text_input = args.text
    elif args.text_file:
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"Text file not found: {args.text_file}")
        with open(args.text_file, 'r', encoding='utf-8', errors='ignore') as f:
            text_input = f.read().strip()

    # Validate media paths
    if args.image is not None and not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.video is not None and not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # Init analyzers
    txt = TextSentiment(device=device)
    img = ImageSentiment(device=device)
    vid = VideoSentiment(device=device, frame_sample_rate=args.video_sample_rate, max_frames=args.video_max_frames)

    # Predict dists
    text_dist = txt.predict_proba(text_input) if text_input else {k: 0.0 for k in LABELS}
    image_dist = img.predict_proba(args.image) if args.image else {k: 0.0 for k in LABELS}
    video_dist = vid.predict_proba(args.video) if args.video else {k: 0.0 for k in LABELS}

    fused = fuse_many([text_dist, image_dist, video_dist], [args.w_text, args.w_image, args.w_video])
    label = argmax_label(fused)

    result = {
        "inputs": {
            "text": text_input,
            "image": args.image,
            "video": args.video
        },
        "proba": {
            "text": text_dist,
            "image": image_dist,
            "video": video_dist,
            "fused": fused
        },
        "prediction": label
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
