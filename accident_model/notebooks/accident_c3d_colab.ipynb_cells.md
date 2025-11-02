```markdown
This file contains the Colab-ready notebook broken into sequential cells. Paste each cell into a new Colab cell and run.

These cells are tuned for Google Colab Free-tier GPU (small batches, num_workers=0, save outputs to Drive).

Cell 1 (Title + mount Drive)
# Accident detection with C3D — Colab notebook (Colab Free GPU tuned)
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/accident_dataset'
!mkdir -p "{DRIVE_ROOT}"

Cell 2 (Install deps)
# Install dependencies (run once). This might take some minutes.
!pip install -q -r /content/drive/MyDrive/accident_dataset/requirements.txt

Cell 3 (Upload or clone repo to Drive)
# Ensure the repository files are available in DRIVE_ROOT.
# Option A: If you have the repo on GitHub and public:
# !git clone https://github.com/<yourusername>/accident-detection-with-C3D.git /content/accident-detection-with-C3D
# !cp -r /content/accident-detection-with-C3D/* "{DRIVE_ROOT}/"
# Option B: Upload the code directory into your Drive via the UI and ensure it lives under DRIVE_ROOT.

Cell 4 (Prepare dataset on Drive)
# Ensure the raw videos are under DRIVE_ROOT/raw_videos/<label or nested directories>
# Example: DRIVE_ROOT/raw_videos/accident/*.mp4 and DRIVE_ROOT/raw_videos/normal/*.mp4

Cell 5 (Extract frames from raw videos)
# This extracts frames into DRIVE_ROOT/frames. Use smaller sample_fps to reduce storage if needed.
!python preprocessing/extract_frames.py --input_dir "{DRIVE_ROOT}/raw_videos" --output_dir "{DRIVE_ROOT}/frames" --ext mp4 --sample_fps 15

Cell 6 (Generate clips and metadata)
# Generate clips into DRIVE_ROOT/clips using 16-frame windows and stride 8.
# This step will create metadata.json in the out folder.
!python preprocessing/generate_clips.py --frames_root "{DRIVE_ROOT}/frames" --out_root "{DRIVE_ROOT}/clips" --frames_per_clip 16 --stride 8 --width 112 --height 112

Cell 7 (Inspect counts)
# Quick inspect the metadata counts to confirm.
python - <<'PY'
import json, os
p = "/content/drive/MyDrive/accident_dataset/clips/metadata.json"
if not os.path.exists(p):
    print("metadata not found at", p)
else:
    m = json.load(open(p))
    print("Train clips:", len(m.get("train",[])))
    print("Val clips:", len(m.get("val",[])))
PY

Cell 8 (Train - use Drive path for out_dir so checkpoints persist)
# Recommended defaults for Colab Free GPU:
# --batch_size 4 --num_workers 0 --epochs 12 --accumulate 1
!python train.py --metadata "{DRIVE_ROOT}/clips/metadata.json" --out_dir "{DRIVE_ROOT}/checkpoints" --epochs 12 --batch_size 4 --num_workers 0 --accumulate 1

Cell 9 (Confirm checkpoints written to Drive)
# List checkpoint files
!ls -lh "{DRIVE_ROOT}/checkpoints"

Cell 10 (Inference on a sample video folder)
# Use the saved best_model.pth stored on Drive for inference on a frames folder.
!python inference.py --checkpoint "{DRIVE_ROOT}/checkpoints/best_model.pth" --video_frames "{DRIVE_ROOT}/frames/<some_video_folder>" --out_json "{DRIVE_ROOT}/inference_results/some_video.json"

Cell 11 (View inference results)
python - <<'PY'
import json
p = "/content/drive/MyDrive/accident_dataset/inference_results/some_video.json"
import os
if os.path.exists(p):
    print(json.dumps(json.load(open(p)), indent=2))
else:
    print("No inference results at", p)
PY

Notes and tips for running in Colab Free GPU:
- Use small batch sizes (4 or lower) to avoid OOM on free GPUs.
- Set num_workers=0 in DataLoader to avoid fork-related issues in Colab.
- If you need larger effective batch size, use --accumulate to perform gradient accumulation.
- Always set --out_dir to a path inside your mounted Drive (e.g., /content/drive/MyDrive/accident_dataset/checkpoints) so checkpoints persist after the session ends.
- If you get CUDA OOM, lower batch_size or use accumulation 2 with batch_size halved.