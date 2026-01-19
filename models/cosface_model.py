"""
Face Recognition - CosFace ResNet50 Model
==========================================
Wrapper para o modelo ResNet50 com CosFace Loss para extração de embeddings faciais.
"""

import sys
from pathlib import Path
from typing import Union

import torch

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .base import BaseModel
from .architectures import create_resnet50


class CosFaceModel(BaseModel):
    """
    Modelo ResNet50 com CosFace Loss para extração de embeddings faciais.
    
    Características:
        - Arquitetura: ResNet50
        - Loss: CosFace (treinamento)
        - Dimensão do embedding: 512
        - Entrada: Imagens 112x112 RGB
        - Normalização: (0.5, 0.5, 0.5)
    
    Nota:
        Este modelo foi treinado com CosFace Loss, que produz embeddings
        altamente discriminativos para reconhecimento facial.
    """
    
    def __init__(
        self,
        weight_path: Union[str, Path],
        device: torch.device = None,
        embedding_dim: int = 512
    ):
        """
        Inicializa o modelo ResNet50 CosFace.
        
        Args:
            weight_path: Caminho para o arquivo .ckpt com os pesos
            device: Dispositivo para execução (cuda/cpu)
            embedding_dim: Dimensão do embedding (default: 512)
        """
        super().__init__(
            model_name="cosface_resnet50",
            weight_path=weight_path,
            device=device,
            embedding_dim=embedding_dim
        )
    
    def _create_architecture(self) -> torch.nn.Module:
        """
        Cria a arquitetura ResNet50.
        
        Returns:
            Modelo ResNet50
        """
        # Criar ResNet50 sem pesos pré-treinados (vamos carregar os nossos)
        model = create_resnet50(
            embedding_dim=self.embedding_dim,
            pretrained=False
        )
        return model


# ===========================================
# Função de conveniência
# ===========================================
def create_cosface_model(
    weight_path: Union[str, Path],
    device: torch.device = None
) -> CosFaceModel:
    """
    Cria uma instância do modelo ResNet50 CosFace.
    
    Args:
        weight_path: Caminho para os pesos
        device: Dispositivo
        
    Returns:
        Instância de CosFaceModel
    """
    return CosFaceModel(weight_path=weight_path, device=device)