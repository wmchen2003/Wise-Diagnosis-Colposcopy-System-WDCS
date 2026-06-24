import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size: int):
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
            hue=0.02,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
