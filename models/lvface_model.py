import sys
from pathlib import Path
from typing import Union

import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .base import BaseModel
from face_module.LVFace.backbones.vit import VisionTransformer


class LVFaceModel(BaseModel):
    """
    Modelo LVFace (ViT-B) para extração de embeddings faciais.

    Arquitetura: Vision Transformer - vit_b_dp005_mask_005
        - embed_dim: 512, depth: 24, num_heads: 8
        - drop_path_rate: 0.05, mask_ratio: 0.05
        - Entrada: 112x112 RGB
        - Saída base: 512-d | Com TTA: 1024-d

    Nota:
        O forward() deste VisionTransformer retorna tensor direto (não tupla),
        diferente do vit.py do TransFace. Compatível com o BaseModel sem
        necessidade de override.

    Referência:
        LVFace: Progressive Cluster Optimization for Large Vision Models
        in Face Recognition (ICCV 2025 Highlight)
        Treinado em Glint360K.
    """

    def __init__(
        self,
        weight_path: Union[str, Path],
        device: torch.device = None,
        embedding_dim: int = 512,
        use_tta: bool = True
    ):
        super().__init__(
            model_name="lvface_b_glint",
            weight_path=weight_path,
            device=device,
            embedding_dim=embedding_dim,
            use_tta=use_tta
        )

    def _create_architecture(self) -> torch.nn.Module:
        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=self.embedding_dim,
            embed_dim=512,
            depth=24,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.05,
            using_checkpoint=True
        )