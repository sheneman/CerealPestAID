#!/usr/bin/env python3
"""
FACT AI - Classifier (Test Run for InceptionV3)
Evaluates a saved InceptionV3 model (trained with aux logits) on a held-out test set.
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
from torchvision import datasets, transforms, models

# Optional: richer metrics if scikit-learn is installed
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# Suppress EXIF warnings (to mirror training)
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 12 bytes but only got 2.")
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.")

def get_inception_v3(pretrained: bool = True, aux_logits: bool = True):
    """Create InceptionV3 with torchvision version compatibility."""
    try:
        # Newer torchvision API uses weights enums
        from torchvision.models import Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=aux_logits)
    except Exception:
        # Older API
        model = models.inception_v3(pretrained=pretrained, aux_logits=aux_logits)
    return model

def infer_num_classes_from_state(state_dict: dict) -> int | None:
    """Infer num_classes from final fc weight in the checkpoint."""
    # Standard InceptionV3 head is model.fc
    keys_to_try = [
        "fc.weight",                 # canonical
        "classifier.weight",         # rare custom heads
    ]
    for k in keys_to_try:
        if k in state_dict:
            return state_dict[k].shape[0]
    # Fallback: guess from any 2D weight with common in_features (2048 typical)
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2 and v.shape[1] in (2048,):
            return v.shape[0]
    return None

def replace_classifier_head(model: nn.Module, num_classes: int):
    """Replace the final classifier layer to match num_classes."""
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

def build_model(checkpoint_path: str, num_classes_from_data: int | None = None) -> tuple[nn.Module, int]:
    """Load InceptionV3 (aux_logits=True for compatibility), adjust head, and load checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint on CPU
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Training saved either a raw state_dict or a full dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    num_classes = infer_num_classes_from_state(state_dict)
    if num_classes is None:
        if num_classes_from_data is None:
            raise RuntimeError(
                "Could not infer num_classes from checkpoint and no dataset class count provided."
            )
        num_classes = num_classes_from_data

    # Keep aux_logits=True so keys like 'AuxLogits.*' can load cleanly
    model = get_inception_v3(pretrained=False, aux_logits=True)
    replace_classifier_head(model, num_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")

    model.eval()  # In eval mode, Inception returns main logits (aux is ignored)
    return model, num_classes

def make_dataloader(test_dir: str, batch_size: int, num_workers: int):
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
        shuffle=False,
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
    all_paths  = [p for p, _ in loader.dataset.samples]  # capture order once

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)

        # In eval mode with aux_logits=True, torchvision typically returns Tensor (main logits).
        # Be defensive in case of older versions returning a tuple.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        logits = outputs.detach().cpu()
        preds = torch.argmax(logits, dim=1)

        all_logits.append(logits)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_preds  = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Overall accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\n=== Test Results (InceptionV3) ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Overall Accuracy: {acc:.4f}")

    # Optional detailed metrics
    if HAVE_SKLEARN:
        print("\nPer-class Precision / Recall / F1:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        print("Confusion Matrix (rows=true, cols=pred):")
        header = "pred→  " + "  ".join([f"{c[:8]:>8}" for c in class_names])
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join([f"{v:>8d}" for v in row])
            print(f"{class_names[i][:8]:>6} | {row_str}")
    else:
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
    p = argparse.ArgumentParser(description="Test InceptionV3 classifier on a test split.")
    p.add_argument("--test-dir", default="./data/labels/test", help="Path to test split (ImageFolder).")
    p.add_argument("--checkpoint", default="./models/best_factai_inceptionv3.pth",
                   help="Path to saved checkpoint (.pth).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for test.")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--csv-out", default="./models/test_predictions_inceptionv3.csv",
                   help="Optional CSV path for per-image predictions (set empty to skip).")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Build dataloader first so we know dataset classes as a fallback
    test_loader, class_names = make_dataloader(args.test_dir, args.batch_size, args.num_workers)
    if len(class_names) == 0:
        raise RuntimeError(f"No classes found in test dir: {args.test_dir}")

    # Build & load model (keeps aux_logits=True for state_dict compatibility)
    model, num_classes = build_model(args.checkpoint, num_classes_from_data=len(class_names))
    if num_classes != len(class_names):
        print(f"[warn] Checkpoint num_classes={num_classes} differs from dataset classes={len(class_names)}. "
              f"Proceeding with checkpoint head; ensure your test set matches the trained classes.")

    # Evaluate
    csv_out = args.csv_out if (args.csv_out and len(args.csv_out.strip()) > 0) else None
    evaluate(model, test_loader, device, class_names, save_csv=csv_out)

if __name__ == "__main__":
    main()

