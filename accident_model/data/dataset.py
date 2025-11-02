
import os
from pathlib import Path
import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np

class AccidentClipDataset(Dataset):
    def __init__(self, metadata_path, transform=None, classes=None):

        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        # metadata is expected to be {"train": [...], "val": [...]} or a list
        if isinstance(self.metadata, dict) and "train" in self.metadata:
            # we will allow selecting subset later; dataset wrapper simple loads list
            # flatten both train and val if needed
            entries = []
            for k in ("train", "val", "test"):
                if k in self.metadata:
                    entries.extend(self.metadata[k])
        elif isinstance(self.metadata, list):
            entries = self.metadata
        else:
            # maybe file pointing to a list for one split
            raise ValueError("Unsupported metadata format. Provide metadata.json with 'train'/'val' lists or a list")
        self.entries = entries
        # build classes
        if classes:
            self.classes = classes
        else:
            labels = sorted(list({e["label"] for e in self.entries}))
            self.classes = {l:i for i,l in enumerate(labels)}
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path = Path(entry["path"])
        arr = np.load(str(path))  # (C,T,H,W) float32 in 0..1
        # apply dataset normalization consistent with C3D baseline:
        # typically subtract mean [0.485,0.456,0.406] and divide by std [0.229,0.224,0.225]
        # but C3D original used mean pixel values (we'll use imagenet-like values for simplicity)
        arr = (arr - np.array([0.485,0.456,0.406]).reshape(3,1,1,1)) / np.array([0.229,0.224,0.225]).reshape(3,1,1,1)
        tensor = torch.from_numpy(arr).float()
        label = self.classes[entry["label"]]
        return tensor, label

def split_metadata_for_loader(metadata_path, split="train"):
    """Return a path-like JSON (list) for a split by reading metadata.json and extracting requested split."""
    mp = Path(metadata_path)
    with open(mp, "r") as f:
        meta = json.load(f)
    if split not in meta:
        raise ValueError(f"Split {split} not in metadata")
    outp = mp.parent / f"meta_{split}.json"
    with open(outp, "w") as fo:
        json.dump(meta[split], fo, indent=2)
    return outp

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    p = split_metadata_for_loader(args.metadata, args.split)
    print("Wrote", p)