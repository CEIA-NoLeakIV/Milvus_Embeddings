import sys
from pathlib import Path
from typing import Union

import torch

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .base import BaseModel
from .architectures import mobilenet_v3_large


class MobileNetModel(BaseModel):
    """
    Modelo MobileNetV3 Large para extração de embeddings faciais.
    
    Características:
        - Arquitetura: MobileNetV3 Large
        - Dimensão do embedding base: 512
        - Com TTA: 1024 dimensões (512 original + 512 flipped)
        - Entrada: Imagens 112x112 RGB
        - Normalização: (0.5, 0.5, 0.5)
    """
    
    def __init__(
        self,
        weight_path: Union[str, Path],
        device: torch.device = None,
        embedding_dim: int = 512,
        use_tta: bool = True
    ):
        """
        Inicializa o modelo MobileNetV3 Large.
        
        Args:
            weight_path: Caminho para o arquivo .ckpt com os pesos
            device: Dispositivo para execução (cuda/cpu)
            embedding_dim: Dimensão do embedding base (default: 512)
            use_tta: Se True, usa TTA e retorna 1024 dims (default: True)
        """
        super().__init__(
            model_name="mobilenetv3_large",
            weight_path=weight_path,
            device=device,
            embedding_dim=embedding_dim,
            use_tta=use_tta
        )
    
    def _create_architecture(self) -> torch.nn.Module:
        """
        Cria a arquitetura MobileNetV3 Large.
        
        Returns:
            Modelo MobileNetV3 Large
        """
        model = mobilenet_v3_large(embedding_dim=self.embedding_dim)
        return model


# ===========================================
# Função de conveniência
# ===========================================
def create_mobilenet_model(
    weight_path: Union[str, Path],
    device: torch.device = None,
    use_tta: bool = True
) -> MobileNetModel:
    """
    Cria uma instância do modelo MobileNetV3 Large.
    
    Args:
        weight_path: Caminho para os pesos
        device: Dispositivo
        use_tta: Se True, usa TTA (1024 dims)
        
    Returns:
        Instância de MobileNetModel
    """
    return MobileNetModel(weight_path=weight_path, device=device, use_tta=use_tta)