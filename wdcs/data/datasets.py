from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SEMANTIC_FEATURES = [
    "acetowhite_epithelium",
    "lesion_margin",
    "punctation",
    "mosaic",
    "atypical_vessels",
    "inner_border_sign",
]


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return mask.astype(np.int64)


class SemanticSignDataset(Dataset):
    """Dataset for six-task semantic-sign recognition."""

    def __init__(self, dataframe, transform=None, feature_names=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.feature_names = feature_names or SEMANTIC_FEATURES

        required = ["image_path"] + self.feature_names
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns for semantic dataset: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = _read_image(row["image_path"])

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        labels = {
            name: torch.tensor(int(row[name]), dtype=torch.long)
            for name in self.feature_names
        }
        return {
            "image": image,
            "labels": labels,
            "image_path": str(row["image_path"]),
        }


class SegmentationDataset(Dataset):
    """Dataset for lesion segmentation. Samples without mask_path are skipped before construction."""

    def __init__(self, dataframe, transform=None):
        if "mask_path" not in dataframe.columns:
            raise ValueError("metadata must contain mask_path for segmentation training")
        df = dataframe.dropna(subset=["mask_path"]).copy()
        df = df[df["mask_path"].astype(str).str.len() > 0]
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = _read_image(row["image_path"])
        mask = _read_mask(row["mask_path"])

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)
        return {
            "image": image,
            "mask": mask,
            "image_path": str(row["image_path"]),
        }


class ImageOnlyDataset(Dataset):
    """Dataset for feature extraction only. This is not a deployment inference wrapper."""

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = _read_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        item = {
            "image": image,
            "image_path": str(row["image_path"]),
        }
        for col in ["split", "diagnosis_label"]:
            if col in row.index:
                item[col] = row[col]
        return item
