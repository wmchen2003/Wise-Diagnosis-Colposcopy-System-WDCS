import torch
import torch.nn as nn
import torch.nn.functional as F

from wdcs.models.backbones import build_segformer_encoder


class SegFormerDecoder(nn.Module):
    """Minimal SegFormer-style MLP decoder.

    Multi-scale encoder features are linearly projected, resized to the highest-resolution
    feature map, concatenated, and fused into a decoder feature map.
    """

    def __init__(self, in_channels, decoder_dim: int = 256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(ch, decoder_dim, kernel_size=1)
            for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(in_channels), decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.proj):
            feat = proj(feat)
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feat)
        fused = self.fuse(torch.cat(projected, dim=1))
        return fused


class SegFormerSegmentationBranch(nn.Module):
    """SegFormer-style lesion segmentation branch.

    Outputs:
    - mask_logits: used for lesion visualization and lesion-extent delineation
    - decoder_feature: fused decoder feature map
    - spatial_features: decoder-derived lesion spatial feature vector after global average pooling
    """

    def __init__(self, encoder_name: str, num_classes: int, pretrained: bool = True, decoder_dim: int = 256):
        super().__init__()
        self.encoder, in_channels = build_segformer_encoder(
            encoder_name=encoder_name,
            pretrained=pretrained,
        )
        self.decoder = SegFormerDecoder(in_channels=in_channels, decoder_dim=decoder_dim)
        self.mask_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.spatial_dim = decoder_dim

    def forward(self, x):
        image_size = x.shape[-2:]
        features = self.encoder(x)
        decoder_feature = self.decoder(features)
        mask_logits = self.mask_head(decoder_feature)
        mask_logits = F.interpolate(mask_logits, size=image_size, mode="bilinear", align_corners=False)
        spatial_features = F.adaptive_avg_pool2d(decoder_feature, output_size=1).flatten(1)
        return {
            "mask_logits": mask_logits,
            "decoder_feature": decoder_feature,
            "spatial_features": spatial_features,
        }
