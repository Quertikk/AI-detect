"""
Fine-tunes an EfficientNet-B0 (ImageNet-pretrained) classifier to tell apart
real vs. AI-generated/face-swapped face images.

Only the classifier head and the last backbone block are unfrozen, which
keeps training to a few minutes per epoch even on a small (4 GB) GPU.

Usage:
    python training/train.py --data data/faces --epochs 3 --batch-size 32

Writes:
    models/classifier.pt   - trained weights + label mapping
    models/metrics.json    - accuracy / precision / recall / F1 / ROC-AUC on the test split
"""
import argparse
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model() -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv_head.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def build_loaders(data_dir: Path, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(data_dir / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    assert train_ds.classes == ["fake", "real"], f"Unexpected class order: {train_ds.classes}"

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        "valid": DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    }
    return loaders, train_ds.classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds),
        "recall": recall_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "n_samples": len(all_labels),
    }


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/faces")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="models/classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, classes = build_loaders(Path(args.data), args.batch_size)
    print(f"Classes (index order): {classes}  |  fake=0, real=1")

    model = build_model().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["valid"], device)
        elapsed = time.time() - start
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"({elapsed:.0f}s)")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            }, out_path)
            print(f"  -> saved new best checkpoint to {out_path}")

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)
    print("Test metrics (best checkpoint):", json.dumps(test_metrics, indent=2))

    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps({
        "test": test_metrics,
        "dataset": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }, indent=2))
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
