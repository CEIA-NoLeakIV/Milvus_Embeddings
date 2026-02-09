# models/topofr_model.py
import sys
from pathlib import Path
from typing import Union
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .base import BaseModel
# Importa as arquiteturas do repositório TopoFR que ajustamos
from face_module.TopoFR.backbones.iresnet import iresnet50, iresnet100, iresnet200

class TopoFRModel(BaseModel):
    def __init__(
        self,
        weight_path: Union[str, Path],
        device: torch.device = None,
        embedding_dim: int = 512,
        use_tta: bool = True
    ):
        # Detecção automática da arquitetura baseada no nome do arquivo
        self.arch_name = "r50"
        path_str = str(weight_path).lower()
        if "r100" in path_str:
            self.arch_name = "r100"
        elif "r200" in path_str:
            self.arch_name = "r200"

        super().__init__(
            model_name=f"topofr_{self.arch_name}",
            weight_path=weight_path,
            device=device,
            embedding_dim=embedding_dim,
            use_tta=use_tta
        )
    
    def _create_architecture(self) -> torch.nn.Module:
        arch_map = {
            "r50": iresnet50,
            "r100": iresnet100,
            "r200": iresnet200
        }
        return arch_map[self.arch_name](fp16=False)
