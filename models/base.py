from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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
            use_tta: Se True, usa TTA e retorna 1024 dims. Se False, retorna 512 dims.
        """
        self.model_name = model_name
        self.weight_path = Path(weight_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.use_tta = use_tta
        
        # Dimensão final do embedding (com ou sem TTA)
        self.output_dim = embedding_dim * 2 if use_tta else embedding_dim
        
        # Modelo será carregado pelas subclasses
        self.model = None
        
        # Transform padrão para imagens (original)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        # Transform com flip horizontal (para TTA)
        self.transform_flip = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        # Carregar modelo
        self._load_model()
    
    @abstractmethod
    def _create_architecture(self) -> torch.nn.Module:
        """
        Cria a arquitetura do modelo.
        Deve ser implementado pelas subclasses.
        
        Returns:
            Modelo PyTorch
        """
        pass
    
    def _load_model(self):
        """Carrega o modelo e os pesos."""
        # Verificar se arquivo de pesos existe
        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"Arquivo de pesos não encontrado: {self.weight_path}"
            )
        
        # Criar arquitetura
        self.model = self._create_architecture()
        
        # Carregar checkpoint (weights_only=False para compatibilidade com PyTorch 2.6+)
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        
        # Extrair state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remover prefixo 'module.' se existir (de treinos com DataParallel/DDP)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        # Carregar pesos
        self.model.load_state_dict(new_state_dict)
        
        # Mover para device e colocar em modo de avaliação
        self.model.to(self.device)
        self.model.eval()
        
        tta_status = "ON (1024 dims)" if self.use_tta else "OFF (512 dims)"
        print(f"✓ Modelo '{self.model_name}' carregado de: {self.weight_path}")
        print(f"  Device: {self.device}")
        print(f"  TTA: {tta_status}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocessa uma imagem PIL.
        
        Args:
            image: Imagem PIL (RGB)
            
        Returns:
            Tensor preprocessado
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Adicionar dimensão de batch
        tensor = tensor.to(self.device)
        
        return tensor
    
    def extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extrai embedding de um tensor de imagem (sem TTA).
        
        Args:
            image_tensor: Tensor da imagem (1, 3, 112, 112)
            
        Returns:
            Embedding como numpy array (512,)
        """
        with torch.no_grad():
            embedding = self.model(image_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def extract_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Extrai embedding diretamente de uma imagem PIL.
        Usa TTA se self.use_tta=True.
        
        Args:
            image: Imagem PIL
            
        Returns:
            Embedding como numpy array (1024 com TTA, 512 sem TTA)
        """
        if self.use_tta:
            return self.extract_embedding_with_tta(image)
        else:
            tensor = self.preprocess(image)
            return self.extract_embedding(tensor)
    
    def extract_embedding_with_tta(self, image: Image.Image) -> np.ndarray:
        """
        Extrai embedding com TTA (Test-Time Augmentation).
        Concatena embedding original + embedding flipped = 1024 dimensões.
        
        Args:
            image: Imagem PIL
            
        Returns:
            Embedding como numpy array (1024,)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preparar tensores
        img_orig = self.transform(image).unsqueeze(0).to(self.device)
        img_flip = self.transform_flip(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            f_orig = self.model(img_orig)
            f_flip = self.model(img_flip)
            # Concatenar: 512 + 512 = 1024 dimensões
            embedding = torch.cat([f_orig, f_flip], dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def extract_embedding_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extrai embedding de um arquivo de imagem.
        Usa TTA se self.use_tta=True.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Embedding como numpy array (1024 com TTA, 512 sem TTA)
        """
        image = Image.open(image_path).convert('RGB')
        return self.extract_embedding_from_pil(image)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"embedding_dim={self.embedding_dim}, "
            f"output_dim={self.output_dim}, "
            f"use_tta={self.use_tta}, "
            f"device={self.device})"
        )