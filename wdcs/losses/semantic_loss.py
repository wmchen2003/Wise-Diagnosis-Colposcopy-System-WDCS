import torch
import torch.nn as nn


class MultiTaskSemanticLoss(nn.Module):
    """Sum of task-specific cross-entropy losses for semantic-sign recognition."""

    def __init__(self, feature_names, class_weights=None, task_weights=None):
        super().__init__()
        self.feature_names = list(feature_names)
        self.task_weights = task_weights or {name: 1.0 for name in self.feature_names}

        self.losses = nn.ModuleDict()
        for name in self.feature_names:
            weight = None
            if class_weights is not None and name in class_weights:
                weight = torch.tensor(class_weights[name], dtype=torch.float32)
            self.losses[name] = nn.CrossEntropyLoss(weight=weight)

    def to(self, device):
        super().to(device)
        for loss_fn in self.losses.values():
            if getattr(loss_fn, "weight", None) is not None:
                loss_fn.weight = loss_fn.weight.to(device)
        return self

    def forward(self, logits, targets):
        total = 0.0
        loss_dict = {}
        for name in self.feature_names:
            loss = self.losses[name](logits[name], targets[name])
            weighted = self.task_weights[name] * loss
            total = total + weighted
            loss_dict[name] = float(loss.detach().cpu())
        return total, loss_dict


def compute_semantic_class_weights(dataframe, feature_num_classes):
    """Compute inverse-frequency class weights for each semantic-sign task."""
    weights = {}
    for feature_name, num_classes in feature_num_classes.items():
        labels = dataframe[feature_name].dropna().astype(int).to_numpy()
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
        counts = torch.clamp(counts, min=1.0)
        inv = counts.sum() / (num_classes * counts)
        weights[feature_name] = inv.tolist()
    return weights
