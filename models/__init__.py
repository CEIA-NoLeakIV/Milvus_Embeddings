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
    """
    
    # Registro de modelos disponíveis
    # Adicionado mobilenetv3_large_iti usando a classe MobileNetModel
    _registry: Dict[str, Type[BaseModel]] = {
        "mobilenetv3_large": MobileNetModel,
        "mobilenetv3_large_iti": MobileNetModel,
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
        use_cache: bool = True,
        use_tta: bool = None
    ) -> BaseModel:
        """
        Cria ou retorna um modelo pelo nome.
        """
        # Verificar se modelo existe
        if model_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. "
                f"Disponíveis: {available}"
            )
        
        # Usar TTA do Config se não especificado
        if use_tta is None:
            use_tta = Config.USE_TTA
        
        # Chave do cache inclui configuração de TTA
        cache_key = f"{model_name}_tta_{use_tta}"
        
        # Verificar cache
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]
        
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
            device=device,
            use_tta=use_tta
        )
        
        # Adicionar ao cache
        if use_cache:
            cls._cache[cache_key] = model
        
        return model
    
    @classmethod
    def get_available_models(cls) -> list:
        return list(cls._registry.keys())
    
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
    
    @classmethod
    def is_loaded(cls, model_name: str, use_tta: bool = None) -> bool:
        if use_tta is None:
            use_tta = Config.USE_TTA
        cache_key = f"{model_name}_tta_{use_tta}"
        return cache_key in cls._cache
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"model_class deve herdar de BaseModel")
        cls._registry[name] = model_class


__all__ = [
    "BaseModel",
    "MobileNetModel",
    "CosFaceModel",
    "ModelFactory",
    "create_mobilenet_model",
    "create_cosface_model"
]