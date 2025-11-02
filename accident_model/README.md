

# FourthEye
# Accident Detection with C3D (Colab-ready)

This repository adapts a C3D-based violence detection project into an accident detection pipeline that runs in Google Colab and loads video clips from Google Drive.

What's different in this branch:
- Training is adapted to run reliably on Google Colab Free-tier GPUs (reduced default batch size, conservative num_workers, optional gradient-accumulation).
- All checkpoints and the best model are saved directly to Google Drive so they persist after the Colab session ends.

Scope implemented here:
- Full implementation of preprocessing, dataset loader, C3D model, training script, and inference script (steps 1–6).
- Colab-ready notebook content (cell-by-cell) optimized for Colab Free GPU runtime and saving results to Drive.

Assumptions:
- You will mount Google Drive in Colab and use a Drive path as the out_dir for training checkpoints and artifacts.
- Dataset will be uploaded to Drive under the path you configure.
- Use the free-tier Colab GPU: default parameters tuned for limited VRAM.

Top-level layout
- preprocessing/
  - extract_frames.py
  - generate_clips.py
- data/
  - dataset.py
- models/
  - c3d.py
- train.py                       <- tuned for Colab free GPU
- inference.py
- notebooks/
  - accident_c3d_colab.ipynb_cells.md  <- Colab-ready cells optimized for free runtime
- requirements.txt
- README.md (this file)

