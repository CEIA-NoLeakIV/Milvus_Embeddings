from .vit import VisionTransformer


def get_model(name, **kwargs):
    num_features = kwargs.get("num_features", 512)

    if name == "vit_b_dp005_mask_005":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05,
            using_checkpoint=True)
    else:
        raise ValueError(f"Arquitetura '{name}' não suportada no módulo LVFace.")