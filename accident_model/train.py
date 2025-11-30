import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import AccidentClipDataset, split_metadata_for_loader
from models.c3d import C3D
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import time
import json
import matplotlib.pyplot as plt
import itertools

def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    losses = []
    preds = []
    targets = []
    optimizer.zero_grad()
    step = 0
    for i, (x,y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        losses.append(float(loss.item() * accumulation_steps))
        preds.extend(out.detach().argmax(dim=1).cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())
        step += 1
        if (i + 1) % 10 == 0:
            print(f"  [Batch {i+1}/{len(loader)}] avg_loss={np.mean(losses):.4f} ", flush=True)
    # if optimizer has pending grads
    if step % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(losses) if len(losses)>0 else 0.0, accuracy_score(targets, preds) if len(preds)>0 else 0.0

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, return_probs=False):
    model.eval()
    losses = []
    preds = []
    targets = []
    probs = []
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        logits = out.cpu().numpy()
        prob = softmax_numpy(logits)
        pred = np.argmax(logits, axis=1).tolist()
        preds.extend(pred)
        probs.extend(prob.tolist())
        targets.extend(y.cpu().numpy().tolist())
    avg_loss = np.mean(losses) if len(losses)>0 else 0.0
    acc = accuracy_score(targets, preds) if len(preds)>0 else 0.0
    if return_probs:
        return avg_loss, acc, np.array(preds), np.array(targets), np.array(probs)
    return avg_loss, acc, np.array(preds), np.array(targets)

def softmax_numpy(x):
    # x: (N, num_classes)
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def plot_and_save_confusion_matrix(cm, classes, out_path, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(out_path), bbox_inches='tight')
    plt.close()

def evaluate_and_save(model, metadata_path, out_dir, device, criterion, classes_mapping, split="test", batch_size=4, num_workers=0):
    # Choose split: prefer 'test' if present in metadata, otherwise use 'val'
    meta_p = Path(metadata_path)
    with open(meta_p, "r") as f:
        meta = json.load(f)
    if split not in meta:
        if "val" in meta:
            split = "val"
        else:
            # try to find any split key
            possible = [k for k in meta.keys() if isinstance(meta[k], list) and len(meta[k])>0]
            if len(possible) == 0:
                raise ValueError("No suitable split found in metadata to evaluate.")
            split = possible[0]
    # write a temporary split file for loader
    tmp_meta = meta[split]
    tmp_meta_path = meta_p.parent / f"meta_{split}_eval.json"
    with open(tmp_meta_path, "w") as f:
        json.dump(tmp_meta, f, indent=2)
    ds = AccidentClipDataset(tmp_meta_path, classes=classes_mapping)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    print(f"Running evaluation on split '{split}' with {len(ds)} samples")
    loss, acc, preds, targets, probs = eval_epoch(model, loader, criterion, device, return_probs=True)
    # compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary' if len(classes_mapping)==2 else 'weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(targets, probs[:,1]) if probs.shape[1] > 1 else float("nan")
    except Exception:
        roc_auc = float("nan")
    cm = confusion_matrix(targets, preds)
    # save results
    results = {
        "split": split,
        "num_samples": int(len(ds)),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "classes": classes_mapping
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # json results
    res_file = out_dir / "evaluation_report.json"
    with open(res_file, "w") as f:
        json.dump(results, f, indent=2)
    # confusion matrix (npy)
    np.save(str(out_dir / "confusion_matrix.npy"), cm)
    # confusion matrix image
    plot_and_save_confusion_matrix(cm, [k for k,v in sorted(classes_mapping.items(), key=lambda x:x[1])], out_dir / "confusion_matrix.png", normalize=False)
    # normalized image
    plot_and_save_confusion_matrix(cm, [k for k,v in sorted(classes_mapping.items(), key=lambda x:x[1])], out_dir / "confusion_matrix_normalized.png", normalize=True)
    print(f"Saved evaluation report to {res_file}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="metadata.json from generate_clips.py")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test", help="Preferred split to evaluate. Falls back to 'val' if not present in metadata.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4, help="Default lowered for Colab free GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", required=True, help="Directory where checkpoints, logs and evaluation will be saved (set to Drive path in Colab)")
    parser.add_argument("--num_workers", type=int, default=0, help="0 recommended for Colab")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume training from")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # prepare metadata splits (writes meta_train.json, meta_val.json next to metadata path)
    meta_train = split_metadata_for_loader(args.metadata, split=args.split)
    meta_val = split_metadata_for_loader(args.metadata, split=args.val_split)
    # build datasets
    train_ds = AccidentClipDataset(meta_train)
    val_ds = AccidentClipDataset(meta_val, classes=train_ds.classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    # model
    model = C3D(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=str(out_dir / "runs"))
    best_val_loss = 0.42
    best_epoch = 1
    start_epoch = 2
    args.epoch=2
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            # load model / optimizer state
            model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", {})))
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"Warning: could not load optimizer state: {e}")
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            best_epoch = ckpt.get("best_epoch", best_epoch)
            print(f"Resuming from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    # --- End resume logic ---
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=args.accumulate)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | time={(t1-t0):.1f}s")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        # checkpoint
        if epoch % args.save_every == 0:
            ckpt = out_dir / f"epoch_{epoch}.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, str(ckpt))
            print(f"Saved checkpoint {ckpt}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = out_dir / "best_model.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, str(best_path))
            print(f"Saved best model to {best_path}")
    print(f"Training finished. Best val loss {best_val_loss:.4f} at epoch {best_epoch}")
    # save class mapping
    with open(out_dir / "classes.json", "w") as f:
        json.dump(train_ds.classes, f, indent=2)
    writer.close()

    # Final evaluation: prefer 'test' split if present else evaluate on 'val'
    print("Starting final evaluation and saving report to out_dir...")
    eval_results = evaluate_and_save(model, args.metadata, out_dir, device, criterion, train_ds.classes, split=args.test_split, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Evaluation results:", eval_results)

if __name__ == "__main__":
    main()
