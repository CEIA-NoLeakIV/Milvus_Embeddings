from abc import ABC, abstractmethod
from typing import Union, BinaryIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Importar o módulo de pré-processamento centralizado
# O arquivo preprocessing.py deve estar na raiz do projeto
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    preprocess_image,
    extract_embedding_standardized,
    load_image_standardized
)


class BaseModel(ABC):
    """Classe base abstrata para modelos de face recognition."""
    
    def __init__(
        self,
        model_name: str,
        weight_path: Union[str, Path],
        device: torch.device = None,
        embedding_dim: int = 512,
        use_tta: bool = True
    ):
        """
        Inicializa o modelo base.
        
        Args:
            model_name: Nome do modelo
            weight_path: Caminho para o arquivo de pesos
            device: Dispositivo (cuda/cpu)
            embedding_dim: Dimensão do embedding base (512)
            use_tta: Se True, usa TTA e retorna 1024 dims
        """
        self.model_name = model_name
        self.weight_path = Path(weight_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.use_tta = use_tta
        
        # Dimensão final do embedding
        self.output_dim = embedding_dim * 2 if use_tta else embedding_dim
        
        # Modelo será carregado pelas subclasses
        self.model = None
        
        # Carregar modelo
        self._load_model()
    
    @abstractmethod
    def _create_architecture(self) -> torch.nn.Module:
        """Cria a arquitetura do modelo. Implementado pelas subclasses."""
        pass
    
    def _load_model(self):
        """Carrega o modelo e os pesos."""
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {self.weight_path}")
        
        # Criar arquitetura
        self.model = self._create_architecture()
        
        # Carregar checkpoint
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        
        # Extrair state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remover prefixo 'module.' se existir
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        # Carregar pesos
        self.model.load_state_dict(new_state_dict)
        
        # Mover para device e modo avaliação
        self.model.to(self.device)
        self.model.eval()
        
        tta_status = "ON (1024 dims)" if self.use_tta else "OFF (512 dims)"
        print(f"✓ Modelo '{self.model_name}' carregado de: {self.weight_path}")
        print(f"  Device: {self.device}")
        print(f"  TTA: {tta_status}")
    
    # =====================================================
    # MÉTODOS DE EXTRAÇÃO - TODOS USAM PREPROCESSING CENTRALIZADO
    # =====================================================
    
    def extract_embedding_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extrai embedding de um arquivo de imagem.
        USA PREPROCESSING CENTRALIZADO.
        """
        return extract_embedding_standardized(
            self,
            file_path=image_path,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Extrai embedding de uma imagem PIL.
        USA PREPROCESSING CENTRALIZADO.
        """
        return extract_embedding_standardized(
            self,
            pil_image=image,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """
        Extrai embedding de bytes de imagem.
        USA PREPROCESSING CENTRALIZADO.
        """
        return extract_embedding_standardized(
            self,
            file_bytes=file_bytes,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_stream(self, file_stream: BinaryIO) -> np.ndarray:
        """
        Extrai embedding de um stream de arquivo.
        USA PREPROCESSING CENTRALIZADO.
        """
        return extract_embedding_standardized(
            self,
            file_stream=file_stream,
            use_tta=self.use_tta
        )

    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        DEPRECATED: Use extract_embedding_from_pil() diretamente.
        Mantido para compatibilidade.
        """
        return preprocess_image(pil_image=image, device=self.device)
    
    def extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        DEPRECATED: Use extract_embedding_from_*() diretamente.
        Extrai embedding de um tensor já preprocessado.
        """
        with torch.no_grad():
            embedding = self.model(image_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        return embedding
    
    def extract_embedding_with_tta(self, image: Image.Image) -> np.ndarray:
        """
        DEPRECATED: Use extract_embedding_from_pil() com use_tta=True.
        """
        return extract_embedding_standardized(self, pil_image=image, use_tta=True)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"embedding_dim={self.embedding_dim}, "
            f"output_dim={self.output_dim}, "
            f"use_tta={self.use_tta}, "
            f"device={self.device})"
        )