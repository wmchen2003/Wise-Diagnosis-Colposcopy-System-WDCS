import torch
import torch.nn as nn

from wdcs.models.backbones import build_classification_backbone


class SemanticSignRecognitionBranch(nn.Module):
    """Shared backbone with six task-specific semantic-sign heads."""

    def __init__(self, backbone_name: str, feature_num_classes: dict, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.feature_names = list(feature_num_classes.keys())
        self.feature_num_classes = dict(feature_num_classes)

        self.backbone, backbone_dim = build_classification_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone_dim, num_classes),
            )
            for name, num_classes in self.feature_num_classes.items()
        })

    def forward(self, x):
        shared_feature = self.backbone(x)
        logits = {name: head(shared_feature) for name, head in self.heads.items()}
        return logits

    @torch.no_grad()
    def predict_semantic_onehot(self, x):
        """Return concatenated one-hot semantic-sign representation."""
        logits = self.forward(x)
        onehot_list = []
        pred_dict = {}
        for name in self.feature_names:
            pred = logits[name].argmax(dim=1)
            pred_dict[name] = pred
            onehot = torch.nn.functional.one_hot(
                pred,
                num_classes=self.feature_num_classes[name],
            ).float()
            onehot_list.append(onehot)
        z_sem = torch.cat(onehot_list, dim=1)
        return z_sem, pred_dict
