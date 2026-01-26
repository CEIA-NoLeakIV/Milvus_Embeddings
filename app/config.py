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
    # Face Detection (RetinaFace)
    # ===========================================
    # Habilita/desabilita detecção facial automática
    # Quando habilitado, todas as imagens passam por:
    #   1. Detecção de face com RetinaFace
    #   2. Crop da região facial
    #   3. Alinhamento usando landmarks (5 pontos)
    USE_FACE_DETECTION = True
    
    # Modelo do RetinaFace a ser usado
    # Opções: "retinaface_mnet_v2" (mais rápido), "retinaface_r50" (mais preciso)
    FACE_DETECTOR_MODEL = "retinaface_mnet_v2"
    
    # Limiar de confiança para detecção facial (0.0 a 1.0)
    # Faces com score abaixo deste valor são descartadas
    # Valores mais altos = menos falsos positivos, mais falsos negativos
    FACE_DETECTION_CONF_THRESHOLD = 0.35
    
    # Quando múltiplas faces são detectadas, qual selecionar:
    # True = seleciona a MAIOR face (por área do bounding box)
    # False = seleciona a face com MAIOR CONFIANÇA (score)
    FACE_DETECTION_SELECT_LARGEST = True
    
    # Comportamento quando nenhuma face é detectada:
    # "error" = levanta exceção NoFaceDetectedError
    # "fallback" = usa imagem original sem alinhamento (não recomendado)
    FACE_DETECTION_NO_FACE_POLICY = "error"
    
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
    def is_face_detection_available(cls) -> bool:
        """
        Verifica se a detecção facial está disponível.
        
        Returns:
            True se uniface está instalado e funcional
        """
        try:
            from utils.face_detection import is_face_detection_available
            return is_face_detection_available()
        except ImportError:
            return False
    
    @classmethod
    def get_face_detection_config(cls) -> dict:
        """
        Retorna as configurações de detecção facial como dict.
        
        Útil para passar para funções de preprocessing.
        
        Returns:
            Dict com configurações de detecção facial
        """
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
        """
        Aplica as configurações ao módulo de preprocessing.
        
        Deve ser chamado na inicialização da aplicação para
        sincronizar as configurações.
        """
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
        print(f"  COLLECTION:         {cls.COLLECTION_NAME}")
        print(f"  EMBEDDING_DIM:      {cls.EMBEDDING_DIM}")
        print()
        print("  --- Modelos ---")
        print(f"  USE_TTA:            {cls.USE_TTA}")
        print(f"  MODELS:             {cls.AVAILABLE_MODELS}")
        print(f"  DEFAULT_MODEL:      {cls.DEFAULT_MODEL}")
        print()
        print("  --- Face Detection ---")
        fd_available = cls.is_face_detection_available()
        print(f"  AVAILABLE:          {fd_available}")
        print(f"  USE_FACE_DETECTION: {cls.USE_FACE_DETECTION}")
        print(f"  DETECTOR_MODEL:     {cls.FACE_DETECTOR_MODEL}")
        print(f"  CONF_THRESHOLD:     {cls.FACE_DETECTION_CONF_THRESHOLD}")
        print(f"  SELECT_LARGEST:     {cls.FACE_DETECTION_SELECT_LARGEST}")
        print(f"  NO_FACE_POLICY:     {cls.FACE_DETECTION_NO_FACE_POLICY}")
        print("=" * 60)


# ===========================================
# Inicialização automática
# ===========================================
def init_config():
    """
    Inicializa as configurações e aplica ao preprocessing.
    
    Deve ser chamado na inicialização da aplicação.
    """
    # Verificar se detecção facial está disponível
    if Config.USE_FACE_DETECTION:
        if Config.is_face_detection_available():
            print("✓ Face detection enabled and available")
            Config.apply_to_preprocessing()
        else:
            print("⚠️  Face detection enabled but uniface not installed!")
            print("   Install with: pip install uniface")
            print("   Face detection will be disabled.")
            Config.USE_FACE_DETECTION = False


# ===========================================
# Configuração via variáveis de ambiente
# ===========================================
def load_from_env():
    """
    Carrega configurações de variáveis de ambiente.
    
    Variáveis suportadas:
        - FACE_DETECTION_ENABLED: "true" ou "false"
        - FACE_DETECTION_CONF_THRESHOLD: float (0.0 a 1.0)
        - FACE_DETECTION_SELECT_LARGEST: "true" ou "false"
        - USE_TTA: "true" ou "false"
        - EMBEDDING_DIM: int (512 ou 1024)
    """
    # Face Detection
    if os.environ.get("FACE_DETECTION_ENABLED"):
        Config.USE_FACE_DETECTION = os.environ["FACE_DETECTION_ENABLED"].lower() == "true"
    
    if os.environ.get("FACE_DETECTION_CONF_THRESHOLD"):
        Config.FACE_DETECTION_CONF_THRESHOLD = float(os.environ["FACE_DETECTION_CONF_THRESHOLD"])
    
    if os.environ.get("FACE_DETECTION_SELECT_LARGEST"):
        Config.FACE_DETECTION_SELECT_LARGEST = os.environ["FACE_DETECTION_SELECT_LARGEST"].lower() == "true"
    
    # TTA
    if os.environ.get("USE_TTA"):
        Config.USE_TTA = os.environ["USE_TTA"].lower() == "true"
        Config.EMBEDDING_DIM = 1024 if Config.USE_TTA else 512
    
    if os.environ.get("EMBEDDING_DIM"):
        Config.EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])


# Carregar configurações de ambiente automaticamente
load_from_env()