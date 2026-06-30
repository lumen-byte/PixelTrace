"""
PixelTrace - End-to-End CNN Fine-Tuning
----------------------------------------
Fine-tunes MobileNetV3-Small on the screen vs. natural classification task.

Why fine-tuning over frozen features:
  - Frozen MobileNetV3 gives ImageNet features (object recognition) → not optimal
  - Fine-tuning the last 2 conv blocks directly on screen recapture images allows
    the network to learn domain-specific texture and frequency detectors
  - With 200 images + strong augmentation, we effectively see ~6000 training examples
  - Heavy regularisation (dropout=0.5, weight_decay=1e-3, early stopping) prevents overfit

Training:
    python -m ml.finetune_cnn           # 10-fold CV evaluation + final model
    python -m ml.finetune_cnn --quick   # skip CV, just train final model on all data
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import timm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 40
LR            = 3e-4
WEIGHT_DECAY  = 1e-3
DROPOUT       = 0.5
PATIENCE      = 10          # early stopping patience
N_FOLDS       = 10
SEED          = 42

# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    """Loads images from dataset/train and dataset/test directories."""

    CLASSES = {"natural": 0, "screen": 1}

    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform
        for root in root_dirs:
            root = Path(root)
            for cls_name, label in self.CLASSES.items():
                cls_dir = root / cls_name
                if not cls_dir.exists():
                    continue
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    """Strong augmentation for training, deterministic for validation/test."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(device: torch.device) -> nn.Module:
    """
    MobileNetV3-Small with last 2 blocks unfrozen + custom 2-class head.
    Freezing early layers preserves low-level ImageNet features (edges, textures)
    while allowing the network to adapt high-level representations.
    """
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=2,
        drop_rate=DROPOUT,
    )

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 2 feature blocks + classifier
    blocks = list(model.blocks.children()) if hasattr(model, "blocks") else []
    for block in blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Always unfreeze classifier layers
    for name, param in model.named_parameters():
        if any(k in name for k in ["classifier", "conv_head", "bn2"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    return model.to(device)


# ── Training / evaluation ─────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    probs_all, labels_all = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        n += len(labels)
        probs_all.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return total_loss / n, correct / n, np.array(probs_all), np.array(labels_all)


def train_model(model, train_loader, val_loader, device, max_epochs=EPOCHS, patience=PATIENCE):
    """Train with early stopping and cosine annealing LR schedule."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch:3d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(full_dataset, labels, device):
    """10-fold stratified CV evaluation."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n── Fold {fold+1}/{N_FOLDS} ─────────────────")

        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)

        # Override transform for this fold (Dataset has train transforms by default)
        val_ds.dataset.transform = get_transforms(train=False)  # type: ignore

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = build_model(device)
        model = train_model(model, train_loader, val_loader, device)

        # Restore val transforms after training
        val_ds.dataset.transform = get_transforms(train=True)  # type: ignore

        # Re-evaluate with clean val transforms
        val_ds_clean = Subset(ImageDataset(["dataset/train", "dataset/test"], get_transforms(train=False)), val_idx)
        val_loader_clean = DataLoader(val_ds_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        _, fold_acc, _, _ = evaluate(model, val_loader_clean, nn.CrossEntropyLoss(), device)
        fold_accs.append(fold_acc)
        logger.info(f"Fold {fold+1} accuracy: {fold_acc:.4f}")

    logger.info(f"\n{'='*50}")
    logger.info(f"10-Fold CV: mean={np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    logger.info(f"Individual folds: {[f'{a:.4f}' for a in fold_accs]}")
    return fold_accs


# ── Final model training ───────────────────────────────────────────────────────
def train_final_model(full_dataset, device, save_path: Path):
    """Train on all available data and save."""
    logger.info("\nTraining final model on ALL available data...")
    all_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Use a small validation split (10%) for early stopping only
    n = len(full_dataset)
    val_n = max(n // 10, 4)
    indices = np.random.RandomState(SEED).permutation(n)
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds_clean = Subset(ImageDataset(["dataset/train", "dataset/test"], get_transforms(train=False)), val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(device)
    model = train_model(model, train_loader, val_loader, device, max_epochs=EPOCHS + 10, patience=PATIENCE + 5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": IMG_SIZE,
        "classes": {"natural": 0, "screen": 1},
    }, save_path)
    logger.info(f"Final model saved to {save_path}")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip CV, just train final model")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    logger.info(f"Device: {device}")

    # Full dataset with training transforms (augmentation)
    full_ds = ImageDataset(["dataset/train", "dataset/test"], get_transforms(train=True))
    labels = [s[1] for s in full_ds.samples]
    logger.info(f"Total images: {len(full_ds)} (screen={labels.count(1)}, natural={labels.count(0)})")

    if not args.quick:
        run_cv(full_ds, labels, device)

    # Train final model on ALL data
    save_path = Path("ml/models/finetuned_mobilenet.pth")
    t0 = time.time()
    train_final_model(full_ds, device, save_path)
    logger.info(f"Total training time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
