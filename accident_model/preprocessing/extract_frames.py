
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

def extract_video_frames(video_path: Path, out_dir: Path, fps_sample: float = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    saved = 0
    sample_every = 1
    if fps_sample and fps_sample > 0 and src_fps > fps_sample:
        sample_every = int(round(src_fps / fps_sample))
    pbar = tqdm(total=total, desc=f"Extracting {video_path.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            fname = out_dir / f"{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved

def is_video_file(p: Path, ext_filter=None):
    if not p.is_file():
        return False
    ext = p.suffix.lower()
    return (ext_filter is None and ext in ALLOWED_VIDEO_EXTS) or (ext_filter is not None and ext == ext_filter)

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ext_filter = f".{args.ext.lower()}" if args.ext else None
    files = list(input_dir.rglob("*"))
    video_files = [f for f in files if is_video_file(f, ext_filter)]
    print(f"Found {len(video_files)} video files under {input_dir}")
    for v in video_files:
        rel = v.relative_to(input_dir).with_suffix('')
        out = output_dir / rel
        saved = extract_video_frames(v, out, fps_sample=args.sample_fps)
        print(f"{v} -> {saved} frames -> {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to raw videos (can be nested)")
    parser.add_argument("--output_dir", required=True, help="Where to store extracted frames")
    parser.add_argument("--ext", default="mp4", help="Video extension to look for, e.g. mp4")
    parser.add_argument("--sample_fps", type=float, default=None,
                        help="If set, sample frames to this fps (float). If not set, keep original fps.")
    args = parser.parse_args()
    main(args)