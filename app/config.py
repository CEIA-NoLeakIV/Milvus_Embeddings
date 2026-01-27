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
    
    # EMBEDDING_DIM = 1024 (com TTA: 512 original + 512 flipped)
    EMBEDDING_DIM = 1024
    
    # ===========================================
    # TTA (Test-Time Augmentation)
    # ===========================================
    USE_TTA = True  # Se True, usa TTA e gera embeddings de 1024 dims
    
    # ===========================================
    # Face Detection (RetinaFace)
    # ===========================================
    USE_FACE_DETECTION = True
    FACE_DETECTOR_MODEL = "retinaface_mnet_v2"
    FACE_DETECTION_CONF_THRESHOLD = 0.35
    FACE_DETECTION_SELECT_LARGEST = True
    FACE_DETECTION_NO_FACE_POLICY = "error"
    
    # ===========================================
    # Modelos e Collections
    # ===========================================
    AVAILABLE_MODELS = ["mobilenetv3_large", "mobilenetv3_large_iti", "cosface_resnet50"]
    DEFAULT_MODEL = "mobilenetv3_large"
    
    # Mapeamento de modelos para seus pesos
    MODEL_WEIGHTS = {
        "mobilenetv3_large": WEIGHTS_DIR / "mobilenetv3_large.ckpt",
        "mobilenetv3_large_iti": WEIGHTS_DIR / "mobilenetv3_large_iti.ckpt",
        "cosface_resnet50": WEIGHTS_DIR / "resnet50_cosface.ckpt"
    }
    
    # Mapeamento de modelos para suas collections
    MODEL_COLLECTIONS = {
        "mobilenetv3_large": "face_embeddings_mobilenetv3",
        "mobilenetv3_large_iti": "face_embeddings_mobilenetv3_iti",
        "cosface_resnet50": "face_embeddings_cosface"
    }
    
    # Nome padrão da collection (para compatibilidade)
    COLLECTION_NAME = "face_embeddings_mobilenetv3"
    
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
    
    # ===========================================
    # Métodos de classe
    # ===========================================
    
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
    def get_collection_name(cls, model_name: str) -> str:
        """Retorna o nome da collection para um modelo."""
        return cls.MODEL_COLLECTIONS.get(model_name, cls.COLLECTION_NAME)
    
    @classmethod
    def is_face_detection_available(cls) -> bool:
        """Verifica se a detecção facial está disponível."""
        try:
            from utils.face_detection import is_face_detection_available
            return is_face_detection_available()
        except ImportError:
            return False
    
    @classmethod
    def get_face_detection_config(cls) -> dict:
        """Retorna as configurações de detecção facial como dict."""
        return {
            "enabled": cls.USE_FACE_DETECTION,
            "model": cls.FACE_DETECTOR_MODEL,
            "conf_threshold": cls.FACE_DETECTION_CONF_THRESHOLD,
            "select_largest": cls.FACE_DETECTION_SELECT_LARGEST,
            "no_face_policy": cls.FACE_DETECTION_NO_FACE_POLICY,
            "available": cls.is_face_detection_available()
        }
    
    @classmethod
    def apply_to_preprocessing(cls):
        """Aplica as configurações ao módulo de preprocessing."""
        try:
            from preprocessing import set_face_detection_defaults
            set_face_detection_defaults(
                enabled=cls.USE_FACE_DETECTION,
                conf_threshold=cls.FACE_DETECTION_CONF_THRESHOLD,
                select_largest=cls.FACE_DETECTION_SELECT_LARGEST
            )
            print("✓ Configurações de detecção facial aplicadas ao preprocessing")
        except ImportError as e:
            print(f"⚠️  Não foi possível aplicar configurações ao preprocessing: {e}")
    
    @classmethod
    def print_config(cls):
        """Imprime as configurações atuais."""
        print("=" * 60)
        print("       CONFIGURAÇÕES")
        print("=" * 60)
        print(f"  ROOT_DIR:           {cls.ROOT_DIR}")
        print(f"  DATA_DIR:           {cls.DATA_DIR}")
        print(f"  WEIGHTS_DIR:        {cls.WEIGHTS_DIR}")
        print(f"  DEVICE:             {cls.DEVICE}")
        print()
        print("  --- Milvus ---")
        print(f"  MILVUS_DB:          {cls.MILVUS_DB_PATH}")
        print(f"  EMBEDDING_DIM:      {cls.EMBEDDING_DIM}")
        print()
        print("  --- Modelos e Collections ---")
        print(f"  USE_TTA:            {cls.USE_TTA}")
        print(f"  MODELS:             {cls.AVAILABLE_MODELS}")
        print(f"  DEFAULT_MODEL:      {cls.DEFAULT_MODEL}")
        for model, collection in cls.MODEL_COLLECTIONS.items():
            print(f"    - {model}: {collection}")
        print()
        print("  --- Face Detection ---")
        fd_available = cls.is_face_detection_available()
        print(f"  AVAILABLE:          {fd_available}")
        print(f"  USE_FACE_DETECTION: {cls.USE_FACE_DETECTION}")
        print(f"  DETECTOR_MODEL:     {cls.FACE_DETECTOR_MODEL}")
        print("=" * 60)


# ===========================================
# Inicialização automática
# ===========================================
def init_config():
    """Inicializa as configurações e aplica ao preprocessing."""
    if Config.USE_FACE_DETECTION:
        if Config.is_face_detection_available():
            print("✓ Face detection enabled and available")
            Config.apply_to_preprocessing()
        else:
            print("⚠️  Face detection enabled but uniface not installed!")
            Config.USE_FACE_DETECTION = False


# ===========================================
# Configuração via variáveis de ambiente
# ===========================================
def load_from_env():
    """Carrega configurações de variáveis de ambiente."""
    if os.environ.get("FACE_DETECTION_ENABLED"):
        Config.USE_FACE_DETECTION = os.environ["FACE_DETECTION_ENABLED"].lower() == "true"
    
    if os.environ.get("FACE_DETECTION_CONF_THRESHOLD"):
        Config.FACE_DETECTION_CONF_THRESHOLD = float(os.environ["FACE_DETECTION_CONF_THRESHOLD"])
    
    if os.environ.get("FACE_DETECTION_SELECT_LARGEST"):
        Config.FACE_DETECTION_SELECT_LARGEST = os.environ["FACE_DETECTION_SELECT_LARGEST"].lower() == "true"
    
    if os.environ.get("USE_TTA"):
        Config.USE_TTA = os.environ["USE_TTA"].lower() == "true"
        Config.EMBEDDING_DIM = 1024 if Config.USE_TTA else 512
    
    if os.environ.get("EMBEDDING_DIM"):
        Config.EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])


load_from_env()