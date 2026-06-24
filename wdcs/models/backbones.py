import timm


def build_classification_backbone(backbone_name: str, pretrained: bool = True):
    """Build a timm classification backbone that returns a global feature vector."""
    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim = model.num_features
    return model, feature_dim


def build_segformer_encoder(encoder_name: str, pretrained: bool = True, out_indices=(0, 1, 2, 3)):
    """Build a MiT/SegFormer-style feature encoder through timm.

    Typical names depend on the installed timm version, for example `mit_b0` or `mit_b1`.
    """
    encoder = timm.create_model(
        encoder_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=out_indices,
    )
    channels = encoder.feature_info.channels()
    return encoder, channels
