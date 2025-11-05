
import argparse
from pathlib import Path
import numpy as np
import torch
import json
from models.c3d import C3D
from preprocessing.generate_clips import sliding_clips_from_folder


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def load_checkpoint(checkpoint_path, device, num_classes=2):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = C3D(num_classes=num_classes)
    # ckpt may contain dict with 'model_state'
    model_state = ckpt.get("model_state", ckpt)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def run_inference_on_folder(model, folder, device, frames_per_clip=16, stride=8, width=112, height=112, smoothing_window=3, threshold=0.6):
    clips = sliding_clips_from_folder(folder, frames_per_clip=frames_per_clip, stride=stride, resize=(width,height))
    results = []
    confidences = []
    timestamps = []
    for start, clip in clips:
        arr = clip.astype(np.float32) / 255.0
        arr = np.transpose(arr, (3,0,1,2))
        # normalization same as dataset
        arr = (arr - np.array([0.485,0.456,0.406]).reshape(3,1,1,1)) / np.array([0.229,0.224,0.225]).reshape(3,1,1,1)
        tensor = torch.from_numpy(arr).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(tensor)
            out = out.cpu().numpy().squeeze()
            prob = softmax(out)[1]  # assuming class 1 is 'accident'
        confidences.append(float(prob))
        timestamps.append(int(start))
        results.append({"start_frame": int(start), "confidence": float(prob)})
    # smoothing (simple moving average)
    conf_arr = np.array(confidences)
    if len(conf_arr) == 0:
        return {"predictions": [], "events": []}
    kernel = np.ones(smoothing_window)/smoothing_window
    smoothed = np.convolve(conf_arr, kernel, mode='same')
    events = []
    active = False
    current_event = None
    for i, s in enumerate(smoothed):
        if s >= threshold and not active:
            active = True
            current_event = {"start_frame": int(timestamps[i]), "confidence": float(s), "end_frame": int(timestamps[i])}
        elif s >= threshold and active:
            current_event["end_frame"] = int(timestamps[i])
            current_event["confidence"] = max(current_event["confidence"], float(s))
        elif s < threshold and active:
            events.append(current_event)
            active = False
            current_event = None
    if active and current_event:
        events.append(current_event)
    return {"predictions": results, "smoothed": smoothed.tolist(), "events": events}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_frames", required=True, help="Folder with frames 00000.jpg ...")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    out = run_inference_on_folder(model, Path(args.video_frames), device, frames_per_clip=args.frames_per_clip, stride=args.stride, width=args.width, height=args.height, smoothing_window=args.smoothing, threshold=args.threshold)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote inference results to {args.out_json}")
    print("Detected events:", out.get("events", []))

if __name__ == "__main__":
    main()
