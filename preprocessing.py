import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, BinaryIO, Tuple

import torch
from PIL import Image
from torchvision import transforms


# ===========================================
# Importação do módulo de detecção facial
# ===========================================
try:
    from utils.face_detection import (
        detect_and_align_face,
        is_face_detection_available,
        NoFaceDetectedError,
        FaceDetector
    )
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    NoFaceDetectedError = Exception  # Fallback
    print("⚠️  Warning: Face detection module not available.")
    print("   Face detection will be disabled.")


# ===========================================
# Configurações de Pré-processamento
# ===========================================
IMAGE_SIZE = (112, 112)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
INTERPOLATION = Image.BILINEAR  # Padronizar interpolação

# Configurações de detecção facial (defaults)
# Podem ser sobrescritas pelo Config do app
DEFAULT_USE_FACE_DETECTION = True
DEFAULT_FACE_CONF_THRESHOLD = 0.35
DEFAULT_SELECT_LARGEST_FACE = True


# ===========================================
# Transform ÚNICO (usado em TODO lugar)
# ===========================================
_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

_transform_flip = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=1.0),  # Sempre faz flip
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# Transform para imagens já alinhadas (sem resize, já estão em 112x112)
_transform_aligned = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

_transform_aligned_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


# ===========================================
# Função de Carregamento Padronizada
# ===========================================
def load_image_standardized(
    file_path: Optional[Union[str, Path]] = None,
    pil_image: Optional[Image.Image] = None,
    file_bytes: Optional[bytes] = None,
    file_stream: Optional[BinaryIO] = None
) -> Image.Image:
    """
    Carrega imagem de qualquer fonte e retorna PIL Image RGB padronizada.
    
    IMPORTANTE: Apenas UMA das fontes deve ser fornecida.
    
    Args:
        file_path: Caminho para arquivo de imagem
        pil_image: Imagem PIL já carregada
        file_bytes: Bytes da imagem (ex: uploaded_file.read())
        file_stream: Stream de arquivo (ex: flask request.files['image'].stream)
    
    Returns:
        PIL Image no modo RGB
    
    Raises:
        ValueError: Se nenhuma ou mais de uma fonte for fornecida
    """
    # Validar que apenas uma fonte foi fornecida
    sources = [file_path, pil_image, file_bytes, file_stream]
    provided = sum(1 for s in sources if s is not None)
    
    if provided == 0:
        raise ValueError("Deve fornecer uma fonte de imagem")
    if provided > 1:
        raise ValueError("Deve fornecer apenas UMA fonte de imagem")
    
    # Carregar de acordo com a fonte
    if file_path is not None:
        # Carregar do disco
        image = Image.open(file_path)
    
    elif pil_image is not None:
        # Já é PIL Image
        image = pil_image
    
    elif file_bytes is not None:
        # Carregar de bytes
        image = Image.open(io.BytesIO(file_bytes))
    
    elif file_stream is not None:
        # Carregar de stream
        # Ler todo o conteúdo do stream para garantir consistência
        content = file_stream.read()
        # Resetar stream caso precise ser usado novamente
        if hasattr(file_stream, 'seek'):
            file_stream.seek(0)
        image = Image.open(io.BytesIO(content))
    
    # SEMPRE converter para RGB (remove canal alpha, converte grayscale, etc)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


# ===========================================
# Função de Detecção e Alinhamento Facial
# ===========================================
def detect_and_align_face_from_pil(
    pil_image: Image.Image,
    conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
    select_largest: bool = DEFAULT_SELECT_LARGEST_FACE
) -> Image.Image:
    """
    Detecta e alinha face de uma imagem PIL.
    
    Args:
        pil_image: Imagem PIL RGB
        conf_threshold: Limiar de confiança do detector
        select_largest: Se True, seleciona maior face; senão, mais confiante
    
    Returns:
        PIL Image com face alinhada (112x112 RGB)
    
    Raises:
        NoFaceDetectedError: Se nenhuma face for detectada
        RuntimeError: Se detecção facial não estiver disponível
    """
    if not FACE_DETECTION_AVAILABLE:
        raise RuntimeError(
            "Face detection not available. "
            "Install uniface: pip install uniface"
        )
    
    # Converter PIL para numpy (RGB)
    image_np = np.array(pil_image)
    
    # Detectar e alinhar
    aligned_np = detect_and_align_face(
        image_np,
        image_size=112,
        conf_threshold=conf_threshold,
        select_largest=select_largest
    )
    
    # Converter de volta para PIL
    return Image.fromarray(aligned_np)


def load_and_align_face(
    file_path: Optional[Union[str, Path]] = None,
    pil_image: Optional[Image.Image] = None,
    file_bytes: Optional[bytes] = None,
    file_stream: Optional[BinaryIO] = None,
    use_face_detection: bool = True,
    conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
    select_largest: bool = DEFAULT_SELECT_LARGEST_FACE
) -> Tuple[Image.Image, bool]:
    """
    Carrega imagem e opcionalmente detecta/alinha a face.
    
    Esta função combina carregamento + detecção em uma única operação.
    
    Args:
        file_path: Caminho para arquivo
        pil_image: Imagem PIL
        file_bytes: Bytes da imagem
        file_stream: Stream de arquivo
        use_face_detection: Se True, aplica detecção + alinhamento
        conf_threshold: Limiar de confiança do detector
        select_largest: Selecionar maior face
    
    Returns:
        Tuple (PIL Image processada, face_detected: bool)
        - Se use_face_detection=True e face encontrada: (imagem alinhada 112x112, True)
        - Se use_face_detection=True e face NÃO encontrada: raises NoFaceDetectedError
        - Se use_face_detection=False: (imagem original RGB, False)
    
    Raises:
        NoFaceDetectedError: Se use_face_detection=True e nenhuma face detectada
    """
    # 1. Carregar imagem
    image = load_image_standardized(
        file_path=file_path,
        pil_image=pil_image,
        file_bytes=file_bytes,
        file_stream=file_stream
    )
    
    # 2. Aplicar detecção facial se habilitado
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        aligned_image = detect_and_align_face_from_pil(
            image,
            conf_threshold=conf_threshold,
            select_largest=select_largest
        )
        return aligned_image, True
    
    elif use_face_detection and not FACE_DETECTION_AVAILABLE:
        print("⚠️  Face detection requested but not available. Using original image.")
        return image, False
    
    else:
        # Sem detecção facial
        return image, False


# ===========================================
# Função de Pré-processamento Principal
# ===========================================
def preprocess_image(
    file_path: Optional[Union[str, Path]] = None,
    pil_image: Optional[Image.Image] = None,
    file_bytes: Optional[bytes] = None,
    file_stream: Optional[BinaryIO] = None,
    return_flipped: bool = False,
    device: torch.device = None,
    use_face_detection: bool = None,
    conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
    select_largest: bool = DEFAULT_SELECT_LARGEST_FACE
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pré-processa imagem de qualquer fonte para tensor pronto para o modelo.
    
    Esta é a função PRINCIPAL que deve ser usada em TODO lugar:
    - populatemilvus.py
    - API Flask
    - Streamlit
    - Scripts de teste
    
    ATUALIZAÇÃO: Agora inclui detecção facial opcional.
    
    Args:
        file_path: Caminho para arquivo
        pil_image: Imagem PIL
        file_bytes: Bytes da imagem
        file_stream: Stream de arquivo
        return_flipped: Se True, retorna também versão flipped (para TTA)
        device: Device para mover o tensor (cuda/cpu)
        use_face_detection: Se True, aplica detecção + crop + alinhamento
                           Se None, usa o default (DEFAULT_USE_FACE_DETECTION)
        conf_threshold: Limiar de confiança do detector facial
        select_largest: Se True, seleciona maior face quando há múltiplas
    
    Returns:
        Se return_flipped=False: tensor (1, 3, 112, 112)
        Se return_flipped=True: tuple (tensor_original, tensor_flipped)
    
    Raises:
        NoFaceDetectedError: Se use_face_detection=True e nenhuma face detectada
    """
    # Determinar se usa detecção facial
    if use_face_detection is None:
        use_face_detection = DEFAULT_USE_FACE_DETECTION
    
    # 1. Carregar e opcionalmente alinhar face
    image, face_was_aligned = load_and_align_face(
        file_path=file_path,
        pil_image=pil_image,
        file_bytes=file_bytes,
        file_stream=file_stream,
        use_face_detection=use_face_detection,
        conf_threshold=conf_threshold,
        select_largest=select_largest
    )
    
    # 2. Escolher transform apropriado
    # Se a face foi alinhada, já está em 112x112, não precisa resize
    if face_was_aligned:
        transform_main = _transform_aligned
        transform_flip = _transform_aligned_flip
    else:
        transform_main = _transform
        transform_flip = _transform_flip
    
    # 3. Aplicar transform
    tensor = transform_main(image)
    
    # 4. Adicionar dimensão de batch
    tensor = tensor.unsqueeze(0)
    
    # 5. Mover para device se especificado
    if device is not None:
        tensor = tensor.to(device)
    
    # 6. Se precisa da versão flipped (para TTA)
    if return_flipped:
        tensor_flip = transform_flip(image).unsqueeze(0)
        if device is not None:
            tensor_flip = tensor_flip.to(device)
        return tensor, tensor_flip
    
    return tensor


# ===========================================
# Extração de Embedding Padronizada
# ===========================================
def extract_embedding_standardized(
    model,
    file_path: Optional[Union[str, Path]] = None,
    pil_image: Optional[Image.Image] = None,
    file_bytes: Optional[bytes] = None,
    file_stream: Optional[BinaryIO] = None,
    use_tta: bool = True,
    use_face_detection: bool = None,
    conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
    select_largest: bool = DEFAULT_SELECT_LARGEST_FACE
) -> np.ndarray:
    """
    Extrai embedding de forma totalmente padronizada.
    
    Esta função garante que o pré-processamento seja IDÊNTICO
    independente da fonte da imagem.
    
    ATUALIZAÇÃO: Agora inclui detecção facial opcional.
    
    Args:
        model: Modelo de face recognition (deve ter .model e .device)
        file_path: Caminho para arquivo
        pil_image: Imagem PIL
        file_bytes: Bytes da imagem
        file_stream: Stream de arquivo
        use_tta: Se True, usa Test-Time Augmentation (concatena original + flip)
        use_face_detection: Se True, aplica detecção + crop + alinhamento
                           Se None, usa o default global
        conf_threshold: Limiar de confiança do detector facial
        select_largest: Selecionar maior face quando há múltiplas
    
    Returns:
        Embedding como numpy array (512 dims sem TTA, 1024 com TTA)
    
    Raises:
        NoFaceDetectedError: Se use_face_detection=True e nenhuma face detectada
    """
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    
    if use_tta:
        # Obter ambos os tensores
        tensor_orig, tensor_flip = preprocess_image(
            file_path=file_path,
            pil_image=pil_image,
            file_bytes=file_bytes,
            file_stream=file_stream,
            return_flipped=True,
            device=device,
            use_face_detection=use_face_detection,
            conf_threshold=conf_threshold,
            select_largest=select_largest
        )
        
        # Extrair embeddings
        with torch.no_grad():
            emb_orig = model.model(tensor_orig)
            emb_flip = model.model(tensor_flip)
            # Concatenar: 512 + 512 = 1024
            embedding = torch.cat([emb_orig, emb_flip], dim=1)
            embedding = embedding.squeeze().cpu().numpy()
    else:
        # Apenas original
        tensor = preprocess_image(
            file_path=file_path,
            pil_image=pil_image,
            file_bytes=file_bytes,
            file_stream=file_stream,
            return_flipped=False,
            device=device,
            use_face_detection=use_face_detection,
            conf_threshold=conf_threshold,
            select_largest=select_largest
        )
        
        with torch.no_grad():
            embedding = model.model(tensor)
            embedding = embedding.squeeze().cpu().numpy()
    
    return embedding


# ===========================================
# Classe Wrapper para Modelos
# ===========================================
class StandardizedModelWrapper:
    """
    Wrapper que força uso do pré-processamento padronizado.
    
    Uso:
        from preprocessing import StandardizedModelWrapper
        from models import ModelFactory
        
        base_model = ModelFactory.create("mobilenetv3_large")
        model = StandardizedModelWrapper(base_model)
        
        # Agora use normalmente:
        embedding = model.extract_embedding_from_path("image.jpg")
        embedding = model.extract_embedding_from_pil(pil_image)
        embedding = model.extract_embedding_from_bytes(file_bytes)
        embedding = model.extract_embedding_from_stream(file_stream)
    """
    
    def __init__(
        self,
        base_model,
        use_tta: bool = None,
        use_face_detection: bool = None,
        conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD
    ):
        """
        Args:
            base_model: Modelo base (MobileNetModel, CosFaceModel, etc)
            use_tta: Sobrescreve configuração de TTA do modelo base
            use_face_detection: Sobrescreve configuração de detecção facial
            conf_threshold: Limiar de confiança do detector
        """
        self.base_model = base_model
        self.model = base_model.model
        self.device = base_model.device
        self.model_name = base_model.model_name
        
        # Usar TTA do modelo base se não especificado
        self.use_tta = use_tta if use_tta is not None else base_model.use_tta
        self.output_dim = 1024 if self.use_tta else 512
        
        # Configurações de detecção facial
        self.use_face_detection = use_face_detection if use_face_detection is not None else DEFAULT_USE_FACE_DETECTION
        self.conf_threshold = conf_threshold
    
    def extract_embedding_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """Extrai embedding de um arquivo."""
        return extract_embedding_standardized(
            self,
            file_path=image_path,
            use_tta=self.use_tta,
            use_face_detection=self.use_face_detection,
            conf_threshold=self.conf_threshold
        )
    
    def extract_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """Extrai embedding de uma imagem PIL."""
        return extract_embedding_standardized(
            self,
            pil_image=image,
            use_tta=self.use_tta,
            use_face_detection=self.use_face_detection,
            conf_threshold=self.conf_threshold
        )
    
    def extract_embedding_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """Extrai embedding de bytes."""
        return extract_embedding_standardized(
            self,
            file_bytes=file_bytes,
            use_tta=self.use_tta,
            use_face_detection=self.use_face_detection,
            conf_threshold=self.conf_threshold
        )
    
    def extract_embedding_from_stream(self, file_stream: BinaryIO) -> np.ndarray:
        """Extrai embedding de um stream de arquivo."""
        return extract_embedding_standardized(
            self,
            file_stream=file_stream,
            use_tta=self.use_tta,
            use_face_detection=self.use_face_detection,
            conf_threshold=self.conf_threshold
        )
    
    def __repr__(self) -> str:
        return (
            f"StandardizedModelWrapper("
            f"model={self.model_name}, "
            f"use_tta={self.use_tta}, "
            f"use_face_detection={self.use_face_detection}, "
            f"output_dim={self.output_dim})"
        )


# ===========================================
# Funções de Verificação
# ===========================================
def verify_preprocessing_consistency(image_path: str) -> dict:
    """
    Verifica se o pré-processamento é consistente entre diferentes fontes.
    
    Args:
        image_path: Caminho para uma imagem de teste
    
    Returns:
        Dict com resultados da verificação
    """
    # Desabilitar detecção facial para este teste
    # (queremos testar apenas a consistência do carregamento)
    
    # Carregar de diferentes formas
    tensor_from_path = preprocess_image(file_path=image_path, use_face_detection=False)
    
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    tensor_from_bytes = preprocess_image(file_bytes=file_bytes, use_face_detection=False)
    
    with open(image_path, 'rb') as f:
        tensor_from_stream = preprocess_image(file_stream=f, use_face_detection=False)
    
    pil_image = Image.open(image_path).convert('RGB')
    tensor_from_pil = preprocess_image(pil_image=pil_image, use_face_detection=False)
    
    # Verificar igualdade
    path_vs_bytes = torch.equal(tensor_from_path, tensor_from_bytes)
    path_vs_stream = torch.equal(tensor_from_path, tensor_from_stream)
    path_vs_pil = torch.equal(tensor_from_path, tensor_from_pil)
    
    all_equal = path_vs_bytes and path_vs_stream and path_vs_pil
    
    return {
        "all_equal": all_equal,
        "path_vs_bytes": path_vs_bytes,
        "path_vs_stream": path_vs_stream,
        "path_vs_pil": path_vs_pil,
        "tensor_shape": list(tensor_from_path.shape),
        "tensor_dtype": str(tensor_from_path.dtype)
    }


def verify_face_detection_available() -> dict:
    """
    Verifica se a detecção facial está disponível e funcional.
    
    Returns:
        Dict com status da detecção facial
    """
    return {
        "available": FACE_DETECTION_AVAILABLE,
        "uniface_installed": FACE_DETECTION_AVAILABLE,
        "default_enabled": DEFAULT_USE_FACE_DETECTION,
        "default_conf_threshold": DEFAULT_FACE_CONF_THRESHOLD,
        "default_select_largest": DEFAULT_SELECT_LARGEST_FACE
    }


# ===========================================
# Configuração Global (pode ser alterada em runtime)
# ===========================================
def set_face_detection_defaults(
    enabled: bool = None,
    conf_threshold: float = None,
    select_largest: bool = None
):
    """
    Altera as configurações globais de detecção facial.
    
    Útil para configurar via Config do app.
    
    Args:
        enabled: Habilitar/desabilitar detecção facial por padrão
        conf_threshold: Novo limiar de confiança padrão
        select_largest: Nova configuração padrão para seleção de face
    """
    global DEFAULT_USE_FACE_DETECTION, DEFAULT_FACE_CONF_THRESHOLD, DEFAULT_SELECT_LARGEST_FACE
    
    if enabled is not None:
        DEFAULT_USE_FACE_DETECTION = enabled
    if conf_threshold is not None:
        DEFAULT_FACE_CONF_THRESHOLD = conf_threshold
    if select_largest is not None:
        DEFAULT_SELECT_LARGEST_FACE = select_largest
    
    print(f"✓ Face detection defaults updated:")
    print(f"  enabled={DEFAULT_USE_FACE_DETECTION}")
    print(f"  conf_threshold={DEFAULT_FACE_CONF_THRESHOLD}")
    print(f"  select_largest={DEFAULT_SELECT_LARGEST_FACE}")


# ===========================================
# Main (para testes)
# ===========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test preprocessing module")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--verify", action="store_true", help="Verify preprocessing consistency")
    parser.add_argument("--face-detection", action="store_true", help="Test face detection")
    parser.add_argument("--no-face-detection", action="store_true", help="Disable face detection")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preprocessing Module - Test")
    print("=" * 60)
    
    # Verificar status da detecção facial
    fd_status = verify_face_detection_available()
    print(f"\nFace Detection Status:")
    print(f"  Available: {fd_status['available']}")
    print(f"  Default enabled: {fd_status['default_enabled']}")
    
    if args.verify:
        print("\nVerificando consistência do pré-processamento...")
        result = verify_preprocessing_consistency(args.image)
        
        print(f"\nResultados:")
        print(f"  Todos iguais: {result['all_equal']}")
        print(f"  Path vs Bytes: {result['path_vs_bytes']}")
        print(f"  Path vs Stream: {result['path_vs_stream']}")
        print(f"  Path vs PIL: {result['path_vs_pil']}")
        print(f"  Shape: {result['tensor_shape']}")
        
        if result['all_equal']:
            print("\n✓ Pré-processamento está 100% consistente!")
        else:
            print("\n✗ ALERTA: Há inconsistências no pré-processamento!")
    
    elif args.face_detection:
        print("\nTestando detecção facial...")
        
        if not FACE_DETECTION_AVAILABLE:
            print("✗ Face detection not available!")
            exit(1)
        
        try:
            # Testar com detecção
            tensor_with_fd = preprocess_image(
                file_path=args.image,
                use_face_detection=True
            )
            print(f"✓ Com detecção facial: {tensor_with_fd.shape}")
            
            # Testar sem detecção
            tensor_without_fd = preprocess_image(
                file_path=args.image,
                use_face_detection=False
            )
            print(f"✓ Sem detecção facial: {tensor_without_fd.shape}")
            
            # Comparar
            are_equal = torch.equal(tensor_with_fd, tensor_without_fd)
            print(f"\nTensores iguais: {are_equal}")
            print("(Esperado: False, pois detecção facial alinha a imagem)")
            
        except NoFaceDetectedError as e:
            print(f"✗ Nenhuma face detectada: {e}")
        except Exception as e:
            print(f"✗ Erro: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Demo de uso
        use_fd = not args.no_face_detection
        print(f"\nDemo de pré-processamento (face_detection={use_fd})")
        print("=" * 50)
        
        try:
            # Carregar de diferentes formas
            tensor = preprocess_image(
                file_path=args.image,
                use_face_detection=use_fd
            )
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor dtype: {tensor.dtype}")
            print(f"Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            
            # Com TTA
            tensor_orig, tensor_flip = preprocess_image(
                file_path=args.image,
                return_flipped=True,
                use_face_detection=use_fd
            )
            print(f"\nCom TTA:")
            print(f"  Original: {tensor_orig.shape}")
            print(f"  Flipped: {tensor_flip.shape}")
            
            print("\n✓ Teste concluído com sucesso!")
            
        except NoFaceDetectedError as e:
            print(f"\n✗ Nenhuma face detectada: {e}")
        except Exception as e:
            print(f"\n✗ Erro: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)