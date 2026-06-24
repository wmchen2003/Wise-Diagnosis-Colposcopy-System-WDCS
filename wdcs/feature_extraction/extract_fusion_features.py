import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wdcs.data.datasets import ImageOnlyDataset
from wdcs.data.transforms import get_val_transform
from wdcs.models.semantic_branch import SemanticSignRecognitionBranch
from wdcs.models.segmentation_branch import SegFormerSegmentationBranch
from wdcs.utils.io import ensure_dir, load_yaml, read_table, set_seed


def build_models(cfg, device):
    sem_ckpt = torch.load(cfg["paths"]["semantic_ckpt"], map_location="cpu")
    seg_ckpt = torch.load(cfg["paths"]["segmentation_ckpt"], map_location="cpu")

    semantic_model = SemanticSignRecognitionBranch(
        backbone_name=sem_ckpt.get("backbone_name", cfg["model"]["semantic_backbone"]),
        feature_num_classes=sem_ckpt.get("feature_num_classes", cfg["semantic_features"]),
        pretrained=False,
    )
    semantic_model.load_state_dict(sem_ckpt["model_state"])
    semantic_model.to(device).eval()

    segmentation_model = SegFormerSegmentationBranch(
        encoder_name=seg_ckpt.get("encoder_name", cfg["model"]["segformer_encoder"]),
        num_classes=seg_ckpt.get("num_classes", cfg["model"]["seg_num_classes"]),
        pretrained=False,
        decoder_dim=seg_ckpt.get("decoder_dim", cfg["model"]["decoder_dim"]),
    )
    segmentation_model.load_state_dict(seg_ckpt["model_state"])
    segmentation_model.to(device).eval()
    return semantic_model, segmentation_model


def extract_split_features(df_split, split_name, cfg, semantic_model, segmentation_model, device):
    dataset = ImageOnlyDataset(df_split, transform=get_val_transform(cfg["data"]["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    rows = []
    semantic_feature_names = list(cfg["semantic_features"].keys())

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract {split_name}"):
            images = batch["image"].to(device, non_blocking=True)

            z_sem, pred_dict = semantic_model.predict_semantic_onehot(images)
            z_seg = segmentation_model(images)["spatial_features"]
            z_fusion = torch.cat([z_sem, z_seg], dim=1)

            z_sem = z_sem.cpu().numpy()
            z_seg = z_seg.cpu().numpy()
            z_fusion = z_fusion.cpu().numpy()

            batch_size = images.size(0)
            for i in range(batch_size):
                row = {
                    "image_path": batch["image_path"][i],
                    "split": split_name,
                }
                if "diagnosis_label" in batch:
                    row["diagnosis_label"] = batch["diagnosis_label"][i]

                for feature_name in semantic_feature_names:
                    row[f"pred_{feature_name}"] = int(pred_dict[feature_name][i].cpu())

                for j, value in enumerate(z_sem[i]):
                    row[f"sem_{j:03d}"] = float(value)
                for j, value in enumerate(z_seg[i]):
                    row[f"seg_{j:03d}"] = float(value)
                for j, value in enumerate(z_fusion[i]):
                    row[f"fusion_{j:03d}"] = float(value)

                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 2026))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_model, segmentation_model = build_models(cfg, device)

    df = read_table(cfg["data"]["metadata_path"])
    split_names = [cfg["data"]["train_split"], cfg["data"]["val_split"]] + cfg["data"].get("test_splits", [])

    features_dir = Path(cfg["paths"]["features_dir"])
    ensure_dir(features_dir)

    all_tables = []
    for split_name in split_names:
        df_split = df[df["split"] == split_name].copy()
        if len(df_split) == 0:
            continue
        table = extract_split_features(df_split, split_name, cfg, semantic_model, segmentation_model, device)
        table.to_csv(features_dir / f"features_{split_name}.csv", index=False)
        all_tables.append(table)

    if all_tables:
        pd.concat(all_tables, axis=0, ignore_index=True).to_csv(features_dir / "features_all.csv", index=False)


if __name__ == "__main__":
    main()
