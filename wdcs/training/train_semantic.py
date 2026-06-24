import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wdcs.data.datasets import SemanticSignDataset
from wdcs.data.transforms import get_train_transform, get_val_transform
from wdcs.losses.semantic_loss import MultiTaskSemanticLoss, compute_semantic_class_weights
from wdcs.models.semantic_branch import SemanticSignRecognitionBranch
from wdcs.utils.io import ensure_dir, load_yaml, read_table, save_json, set_seed


def move_labels_to_device(labels, device):
    return {k: v.to(device, non_blocking=True) for k, v in labels.items()}


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(images)
                loss, _ = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 2026))

    output_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "class_weights")

    df = read_table(cfg["data"]["metadata_path"])
    train_df = df[df["split"] == cfg["data"]["train_split"]].copy()
    val_df = df[df["split"] == cfg["data"]["val_split"]].copy()

    feature_num_classes = cfg["semantic_features"]
    feature_names = list(feature_num_classes.keys())

    class_weights = None
    if cfg["loss"].get("semantic_auto_class_weights", True):
        class_weights = compute_semantic_class_weights(train_df, feature_num_classes)
        save_json(class_weights, cfg["paths"]["class_weights_json"])

    train_ds = SemanticSignDataset(
        train_df,
        transform=get_train_transform(cfg["data"]["image_size"]),
        feature_names=feature_names,
    )
    val_ds = SemanticSignDataset(
        val_df,
        transform=get_val_transform(cfg["data"]["image_size"]),
        feature_names=feature_names,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticSignRecognitionBranch(
        backbone_name=cfg["model"]["semantic_backbone"],
        feature_num_classes=feature_num_classes,
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    criterion = MultiTaskSemanticLoss(feature_names, class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )
    scaler = torch.cuda.amp.GradScaler() if cfg["training"].get("use_amp", True) and device.type == "cuda" else None

    best_val_loss = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss = run_one_epoch(model, val_loader, criterion, optimizer=None, scaler=None, device=device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "feature_num_classes": feature_num_classes,
            "backbone_name": cfg["model"]["semantic_backbone"],
            "val_loss": val_loss,
        }
        torch.save(ckpt, output_dir / "checkpoints" / "semantic_branch_last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, cfg["paths"]["semantic_ckpt"])


if __name__ == "__main__":
    main()
