#!/usr/bin/env python3
"""
FACT AI - Classifier (Test Run)
Evaluates a saved EfficientNet-B6 model on a held-out test set.
Writes metrics to stdout and per-image predictions to CSV.

Luke Sheneman
sheneman@uidaho.edu
"""

import os
import csv
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

# Optional: richer metrics if scikit-learn is installed
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# Suppress specific warnings you already suppress in training
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 12 bytes but only got 2.")
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.")

def build_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    """Load EfficientNet backbone and adapt final layer to checkpoint num_classes, then load weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load on CPU first for portability; we'll .to(device) later
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = state.get('model_state_dict', state)  # allow raw state_dict files

    # Infer num_classes from the saved classifier layer
    if '_fc.weight' in state_dict:
        num_classes = state_dict['_fc.weight'].shape[0]
    else:
        # Fallback if key differs (some EfficientNet forks use 'classifier' or similar)
        fc_keys = [k for k in state_dict.keys() if k.endswith('weight') and 'fc' in k]
        if not fc_keys:
            raise KeyError("Could not infer num_classes from checkpoint. Missing '_fc.weight'.")
        num_classes = state_dict[fc_keys[0]].shape[0]

    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def make_dataloader(test_dir: str, batch_size: int, num_workers: int) -> tuple[DataLoader, list[str]]:
    """Create a deterministic test loader and return class names."""
    transform_test = transforms.Compose([
        transforms.Resize(572),
        transforms.CenterCrop(528),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(root=test_dir, transform=transform_test)
    class_names = test_ds.classes
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,             # deterministic order for test
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader, class_names

@torch.no_grad()
def evaluate(model, loader, device, class_names, save_csv: str | None):
    """Run inference, compute metrics, optionally write per-image predictions CSV."""
    model = model.to(device)
    all_logits = []
    all_preds = []
    all_labels = []
    all_paths  = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        logits = outputs.detach().cpu()
        preds = torch.argmax(logits, dim=1)

        all_logits.append(logits)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu())

    # Gather file paths (need to re-open dataset targets/paths from loader.dataset)
    # loader.dataset is an ImageFolder, with samples = [(path, class_idx), ...]
    for path, _ in loader.dataset.samples:
        all_paths.append(path)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_preds  = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Overall accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\n=== Test Results ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Overall Accuracy: {acc:.4f}")

    # Optional detailed metrics
    if HAVE_SKLEARN:
        print("\nPer-class Precision / Recall / F1:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        # Pretty-print a compact confusion matrix
        print("Confusion Matrix (rows=true, cols=pred):")
        header = "pred→  " + "  ".join([f"{c[:8]:>8}" for c in class_names])
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join([f"{v:>8d}" for v in row])
            print(f"{class_names[i][:8]:>6} | {row_str}")
    else:
        # Per-class accuracy without sklearn
        print("\n(sklearn not installed) Showing per-class accuracy only.")
        class_correct = np.zeros(len(class_names), dtype=np.int64)
        class_total   = np.zeros(len(class_names), dtype=np.int64)
        for y_true, y_pred in zip(all_labels, all_preds):
            class_total[y_true] += 1
            if y_true == y_pred:
                class_correct[y_true] += 1
        for i, cname in enumerate(class_names):
            if class_total[i] > 0:
                print(f"{cname:>20}: acc={class_correct[i]/class_total[i]:.4f}  ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"{cname:>20}: acc=NA (no samples)")

    # Save per-image predictions (optional)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "true_label", "pred_label", "pred_confidence"] + [f"p_{c}" for c in class_names]
            writer.writerow(header)
            for i in range(len(all_labels)):
                writer.writerow([
                    all_paths[i],
                    class_names[all_labels[i]],
                    class_names[all_preds[i]],
                    float(probs[i, all_preds[i]]),
                    *[float(x) for x in probs[i]]
                ])
        print(f"\nPer-image predictions saved to: {save_csv}")

def parse_args():
    p = argparse.ArgumentParser(description="Test EfficientNet-B6 classifier on a test split.")
    p.add_argument("--test-dir", default="./data/labels/test", help="Path to test split (ImageFolder).")
    p.add_argument("--checkpoint", default="./models/best_factai_efficientnet-b6.pth",
                   help="Path to saved checkpoint (.pth).")
    p.add_argument("--model-name", default="efficientnet-b6", help="EfficientNet variant name.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for test.")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--csv-out", default="./models/test_predictions.csv",
                   help="Optional CSV path for per-image predictions (set empty to skip).")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Build dataloader first so we can sanity-check num classes vs checkpoint
    test_loader, class_names = make_dataloader(args.test_dir, args.batch_size, args.num_workers)
    if len(class_names) == 0:
        raise RuntimeError(f"No classes found in test dir: {args.test_dir}")

    # Build & load model
    model = build_model(args.model_name, args.checkpoint)

    # Evaluate
    csv_out = args.csv_out if (args.csv_out and len(args.csv_out.strip()) > 0) else None
    evaluate(model, test_loader, device, class_names, save_csv=csv_out)

if __name__ == "__main__":
    main()

