import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        prob = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.clamp(min=0), num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1)
            prob = prob * valid
            target_onehot = target_onehot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(prob * target_onehot, dims)
        cardinality = torch.sum(prob + target_onehot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Dice loss + weighted cross-entropy loss for segmentation."""

    def __init__(self, class_weights=None, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def to(self, device):
        super().to(device)
        if getattr(self.ce, "weight", None) is not None:
            self.ce.weight = self.ce.weight.to(device)
        return self

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)
