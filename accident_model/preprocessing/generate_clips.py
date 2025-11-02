
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import random
import json
import math

def load_frames_as_rgb(folder: Path, start_idx: int, num: int, resize=(112,112)):
    frames = []
    for i in range(start_idx, start_idx + num):
        fname = folder / f"{i:05d}.jpg"
        if not fname.exists():
            # pad with last frame if missing
            if len(frames) == 0:
                # return black frames
                frames = [np.zeros((resize[1], resize[0], 3), dtype=np.uint8)] * num
                return np.stack(frames, axis=0)
            frames.append(frames[-1])
        else:
            img = cv2.imread(str(fname))
            if img is None:
                img = np.zeros((resize[1], resize[0], 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize)
            frames.append(img)
    return np.stack(frames, axis=0)  # shape: (num, H, W, C)

def sliding_clips_from_folder(folder: Path, frames_per_clip=16, stride=8, resize=(112,112)):
    # Count available frames by checking consecutive indices
    indices = sorted([int(p.stem) for p in folder.glob("*.jpg")])
    if len(indices) == 0:
        return []
    max_index = indices[-1] + 1
    clips = []
    start = 0
    while start + frames_per_clip <= max_index:
        clip = load_frames_as_rgb(folder, start, frames_per_clip, resize)
        clips.append((start, clip))
        start += stride
    # if remaining tail less than frames_per_clip, pad it (create final clip anchored at end)
    if start < max_index and (max_index - frames_per_clip) > 0:
        start = max_index - frames_per_clip
        clip = load_frames_as_rgb(folder, start, frames_per_clip, resize)
        clips.append((start, clip))
    return clips

def save_clips(clips, out_dir: Path, base_name: str, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, (start, clip) in enumerate(clips):
        fname = out_dir / f"{base_name}__clip_{idx:04d}.npy"
        # convert to float32 and arrange to (C, T, H, W) expected by many C3D loaders later
        arr = clip.astype(np.float32) / 255.0  # normalize to 0..1; further normalization applied in dataset loader
        # transpose from (T,H,W,C) -> (C,T,H,W)
        arr = np.transpose(arr, (3,0,1,2))
        np.save(str(fname), arr)
        saved.append(str(fname))
    return saved

def split_videos(video_list, train_ratio=0.75, seed=42):
    random.seed(seed)
    random.shuffle(video_list)
    n = len(video_list)
    n_train = int(round(n * train_ratio))
    train = video_list[:n_train]
    val = video_list[n_train:]
    return train, val

def dataset_mode(args):
    root = Path(args.frames_root)
    out_root = Path(args.out_root)
    label_dirs = [p for p in root.iterdir() if p.is_dir()]
    metadata = {"train": [], "val": []}
    for label_dir in label_dirs:
        label = label_dir.name
        videos = [p for p in label_dir.iterdir() if p.is_dir()]
        train_v, val_v = split_videos(videos, train_ratio=0.75, seed=args.seed)
        for split_name, video_list in (("train", train_v), ("val", val_v)):
            for v in video_list:
                clips = sliding_clips_from_folder(v, frames_per_clip=args.frames_per_clip,
                                                  stride=args.stride, resize=(args.width, args.height))
                out_dir = out_root / split_name / label
                base = v.name
                saved = save_clips(clips, out_dir, base, label)
                for s in saved:
                    metadata[split_name].append({"path": s, "label": label})
    # write metadata files
    meta_file = out_root / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {meta_file}")

def single_video_mode(args):
    folder = Path(args.video_folder)
    out_dir = Path(args.out_root)
    clips = sliding_clips_from_folder(folder, frames_per_clip=args.frames_per_clip,
                                      stride=args.stride, resize=(args.width, args.height))
    base = folder.name
    saved = save_clips(clips, out_dir, base, args.label or "unknown")
    print(f"Saved {len(saved)} clips to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", help="Root of extracted frames with label subdirs", default=None)
    parser.add_argument("--video_folder", help="Single video folder (frames) for clip generation", default=None)
    parser.add_argument("--out_root", required=True, help="Output root for clips")
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()
    if args.frames_root:
        dataset_mode(args)
    elif args.video_folder:
        single_video_mode(args)
    else:
        raise ValueError("Either --frames_root or --video_folder must be provided")

if __name__ == "__main__":
    main()