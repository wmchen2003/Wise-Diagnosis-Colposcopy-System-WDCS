import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wdcs.data.datasets import SegmentationDataset
from wdcs.data.transforms import get_train_transform, get_val_transform
from wdcs.losses.segmentation_loss import DiceCELoss
from wdcs.models.segmentation_branch import SegFormerSegmentationBranch
from wdcs.utils.io import ensure_dir, load_yaml, read_table, set_seed


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                out = model(images)
                loss = criterion(out["mask_logits"], masks)

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

    df = read_table(cfg["data"]["metadata_path"])
    train_df = df[df["split"] == cfg["data"]["train_split"]].copy()
    val_df = df[df["split"] == cfg["data"]["val_split"]].copy()

    train_ds = SegmentationDataset(train_df, transform=get_train_transform(cfg["data"]["image_size"]))
    val_ds = SegmentationDataset(val_df, transform=get_val_transform(cfg["data"]["image_size"]))

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
    model = SegFormerSegmentationBranch(
        encoder_name=cfg["model"]["segformer_encoder"],
        num_classes=cfg["model"]["seg_num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
        decoder_dim=cfg["model"]["decoder_dim"],
    ).to(device)

    criterion = DiceCELoss(
        class_weights=cfg["loss"].get("segmentation_class_weights", None)
    ).to(device)

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
            "encoder_name": cfg["model"]["segformer_encoder"],
            "decoder_dim": cfg["model"]["decoder_dim"],
            "num_classes": cfg["model"]["seg_num_classes"],
            "val_loss": val_loss,
        }
        torch.save(ckpt, output_dir / "checkpoints" / "segmentation_branch_last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, cfg["paths"]["segmentation_ckpt"])


if __name__ == "__main__":
    main()
