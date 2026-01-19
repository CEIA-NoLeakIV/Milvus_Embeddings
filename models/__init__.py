import sys
from pathlib import Path
from typing import Dict, Type, Union

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .base import BaseModel
from .mobilenet_model import MobileNetModel, create_mobilenet_model
from .cosface_model import CosFaceModel, create_cosface_model

# Importar configurações
from app.config import Config


class ModelFactory:
    """
    Factory para criação de modelos de face recognition.
    
    Exemplo:
        >>> model = ModelFactory.create("mobilenetv3_large")
        >>> embedding = model.extract_embedding_from_path("foto.jpg")
    """
    
    # Registro de modelos disponíveis
    _registry: Dict[str, Type[BaseModel]] = {
        "mobilenetv3_large": MobileNetModel,
        "cosface_resnet50": CosFaceModel
    }
    
    # Cache de modelos carregados
    _cache: Dict[str, BaseModel] = {}
    
    @classmethod
    def create(
        cls,
        model_name: str,
        weight_path: Union[str, Path] = None,
        device=None,
        use_cache: bool = True
    ) -> BaseModel:
        """
        Cria ou retorna um modelo pelo nome.
        
        Args:
            model_name: Nome do modelo ('mobilenetv3_large' ou 'cosface_resnet50')
            weight_path: Caminho para os pesos (opcional, usa default se não informado)
            device: Dispositivo (opcional, usa Config.DEVICE se não informado)
            use_cache: Se True, reutiliza modelo já carregado
            
        Returns:
            Instância do modelo
            
        Raises:
            ValueError: Se o modelo não for encontrado
            FileNotFoundError: Se o arquivo de pesos não existir
        """
        # Verificar se modelo existe
        if model_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. "
                f"Disponíveis: {available}"
            )
        
        # Verificar cache
        if use_cache and model_name in cls._cache:
            return cls._cache[model_name]
        
        # Obter caminho dos pesos
        if weight_path is None:
            weight_path = Config.get_model_weight_path(model_name)
        
        # Obter device
        if device is None:
            device = Config.DEVICE
        
        # Criar modelo
        model_class = cls._registry[model_name]
        model = model_class(
            weight_path=weight_path,
            device=device
        )
        
        # Adicionar ao cache
        if use_cache:
            cls._cache[model_name] = model
        
        return model
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Retorna lista de modelos disponíveis.
        
        Returns:
            Lista com nomes dos modelos
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_cache(cls):
        """Limpa o cache de modelos carregados."""
        cls._cache.clear()
    
    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        """
        Verifica se um modelo está carregado no cache.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se o modelo está no cache
        """
        return model_name in cls._cache
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """
        Registra um novo modelo.
        
        Args:
            name: Nome do modelo
            model_class: Classe do modelo (deve herdar de BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"model_class deve herdar de BaseModel, "
                f"recebido: {type(model_class)}"
            )
        cls._registry[name] = model_class


__all__ = [
    # Classes
    "BaseModel",
    "MobileNetModel",
    "CosFaceModel",
    "ModelFactory",
    
    # Funções de conveniência
    "create_mobilenet_model",
    "create_cosface_model"
]