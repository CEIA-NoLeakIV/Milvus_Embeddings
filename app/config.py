"""
Face Recognition API - Configuration
=====================================
Configurações centralizadas do projeto.
"""

import os
from pathlib import Path
import torch


class Config:
    """Configurações globais da aplicação."""
    
    # ===========================================
    # Paths
    # ===========================================
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    WEIGHTS_DIR = ROOT_DIR / "models" / "weights"
    UPLOADS_DIR = ROOT_DIR / "uploads"
    
    # Criar diretórios se não existirem
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    
    # ===========================================
    # Milvus
    # ===========================================
    MILVUS_DB_PATH = str(DATA_DIR / "milvus_face.db")
    COLLECTION_NAME = "face_embeddings"
    
    # EMBEDDING_DIM = 1024 (com TTA: 512 original + 512 flipped)
    EMBEDDING_DIM = 1024
    
    # ===========================================
    # TTA (Test-Time Augmentation)
    # ===========================================
    USE_TTA = True  # Se True, usa TTA e gera embeddings de 1024 dims
    
    # ===========================================
    # Modelos
    # ===========================================
    AVAILABLE_MODELS = ["mobilenetv3_large", "cosface_resnet50"]
    DEFAULT_MODEL = "mobilenetv3_large"
    
    # Mapeamento de modelos para seus pesos
    MODEL_WEIGHTS = {
        "mobilenetv3_large": WEIGHTS_DIR / "mobilenetv3_large.ckpt",
        "cosface_resnet50": WEIGHTS_DIR / "resnet50_cosface.ckpt"
    }
    
    # ===========================================
    # Device (GPU/CPU)
    # ===========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===========================================
    # API
    # ===========================================
    MAX_BATCH_SIZE = 100
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # ===========================================
    # Image Processing
    # ===========================================
    IMAGE_SIZE = (112, 112)
    NORMALIZE_MEAN = (0.5, 0.5, 0.5)
    NORMALIZE_STD = (0.5, 0.5, 0.5)
    
    @classmethod
    def is_allowed_file(cls, filename: str) -> bool:
        """Verifica se a extensão do arquivo é permitida."""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS
    
    @classmethod
    def get_model_weight_path(cls, model_name: str) -> Path:
        """Retorna o caminho do peso para um modelo."""
        if model_name not in cls.MODEL_WEIGHTS:
            raise ValueError(f"Modelo '{model_name}' não encontrado. "
                           f"Disponíveis: {cls.AVAILABLE_MODELS}")
        return cls.MODEL_WEIGHTS[model_name]
    
    @classmethod
    def print_config(cls):
        """Imprime as configurações atuais."""
        print("=" * 50)
        print("       CONFIGURAÇÕES")
        print("=" * 50)
        print(f"  ROOT_DIR:       {cls.ROOT_DIR}")
        print(f"  DATA_DIR:       {cls.DATA_DIR}")
        print(f"  WEIGHTS_DIR:    {cls.WEIGHTS_DIR}")
        print(f"  DEVICE:         {cls.DEVICE}")
        print(f"  MILVUS_DB:      {cls.MILVUS_DB_PATH}")
        print(f"  COLLECTION:     {cls.COLLECTION_NAME}")
        print(f"  EMBEDDING_DIM:  {cls.EMBEDDING_DIM}")
        print(f"  USE_TTA:        {cls.USE_TTA}")
        print(f"  MODELS:         {cls.AVAILABLE_MODELS}")
        print("=" * 50)