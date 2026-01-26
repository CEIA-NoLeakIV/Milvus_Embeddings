import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List

# Importação condicional do uniface
try:
    from uniface import RetinaFace
    UNIFACE_AVAILABLE = True
except ImportError:
    UNIFACE_AVAILABLE = False
    RetinaFace = None

# Importação do skimage para SimilarityTransform
try:
    from skimage.transform import SimilarityTransform
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    SimilarityTransform = None


# ===========================================
# Exceções Customizadas
# ===========================================
class NoFaceDetectedError(Exception):
    """Exceção levantada quando nenhuma face é detectada na imagem."""
    pass


class MultipleFacesDetectedError(Exception):
    """Exceção levantada quando múltiplas faces são detectadas (se não permitido)."""
    pass


# ===========================================
# Constantes de Alinhamento (ArcFace Reference)
# ===========================================
# Pontos de referência para alinhamento facial no padrão ArcFace
# Ordem: olho_esquerdo, olho_direito, nariz, boca_esquerda, boca_direita
REFERENCE_ALIGNMENT = np.array([
    [38.2946, 51.6963],  # Olho esquerdo
    [73.5318, 51.5014],  # Olho direito
    [56.0252, 71.7366],  # Nariz
    [41.5493, 92.3655],  # Canto esquerdo da boca
    [70.7299, 92.2041]   # Canto direito da boca
], dtype=np.float32)


# ===========================================
# Funções de Alinhamento
# ===========================================
def get_reference_alignment(image_size: int = 112) -> np.ndarray:
    """
    Retorna os pontos de referência ajustados para o tamanho da imagem.
    
    Args:
        image_size: Tamanho da imagem de saída (112 ou 128)
    
    Returns:
        Array numpy com os pontos de referência ajustados
    """
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        # Fallback para 112
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    
    alignment = REFERENCE_ALIGNMENT.copy() * ratio
    alignment[:, 0] += diff_x
    
    return alignment


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Estima a matriz de transformação de similaridade para alinhar landmarks.
    
    Args:
        landmark: Array (5, 2) com coordenadas dos landmarks faciais
        image_size: Tamanho da imagem de saída
    
    Returns:
        Matriz de transformação 2x3
    
    Raises:
        AssertionError: Se landmarks não têm shape (5, 2)
        RuntimeError: Se skimage não está disponível
    """
    if not SKIMAGE_AVAILABLE:
        raise RuntimeError(
            "scikit-image é necessário para alinhamento facial. "
            "Instale com: pip install scikit-image"
        )
    
    assert landmark.shape == (5, 2), f"Landmark deve ter shape (5, 2), recebido {landmark.shape}"
    
    # Obter pontos de referência ajustados
    reference = get_reference_alignment(image_size)
    
    # Calcular transformação de similaridade
    transform = SimilarityTransform()
    transform.estimate(landmark, reference)
    
    # Retornar matriz 2x3
    matrix = transform.params[0:2, :]
    
    return matrix


def align_face(
    image: np.ndarray,
    landmark: np.ndarray,
    image_size: int = 112
) -> np.ndarray:
    """
    Alinha a face na imagem usando os landmarks faciais.
    
    Esta função aplica uma transformação de similaridade para:
    1. Rotacionar a face para ficar alinhada
    2. Escalar para o tamanho desejado
    3. Centralizar a face na imagem de saída
    
    Args:
        image: Imagem de entrada (numpy array, BGR ou RGB)
        landmark: Array (5, 2) com coordenadas dos landmarks
        image_size: Tamanho da imagem de saída (padrão: 112)
    
    Returns:
        Imagem alinhada com tamanho (image_size, image_size, 3)
    """
    # Obter matriz de transformação
    M = estimate_norm(landmark, image_size)
    
    # Aplicar transformação
    aligned = cv2.warpAffine(
        image,
        M,
        (image_size, image_size),
        borderValue=0.0
    )
    
    return aligned


# ===========================================
# Classe FaceDetector
# ===========================================
class FaceDetector:
    """
    Detector facial usando RetinaFace (via uniface).
    
    Implementa padrão singleton para evitar carregar o modelo múltiplas vezes.
    
    Attributes:
        model_name: Nome do modelo RetinaFace
        conf_threshold: Limiar de confiança para detecção
        detector: Instância do RetinaFace
    
    Uso:
        detector = FaceDetector(conf_threshold=0.35)
        faces = detector.detect(image)
        aligned = detector.detect_and_align(image)
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Implementa singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "retinaface_mnet_v2",
        conf_threshold: float = 0.35
    ):
        """
        Inicializa o detector facial.
        
        Args:
            model_name: Nome do modelo (retinaface_mnet_v2 ou retinaface_r50)
            conf_threshold: Limiar de confiança (0.0 a 1.0)
        """
        # Evitar reinicialização do singleton
        if FaceDetector._initialized:
            return
        
        if not UNIFACE_AVAILABLE:
            raise ImportError(
                "uniface é necessário para detecção facial. "
                "Instale com: pip install uniface"
            )
        
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.detector = None
        
        # Inicializar detector
        self._init_detector()
        
        FaceDetector._initialized = True
    
    def _init_detector(self):
        """Inicializa o modelo RetinaFace."""
        try:
            # A API do uniface usa confidence_threshold
            self.detector = RetinaFace(
                confidence_threshold=self.conf_threshold
            )
            print(f"✓ FaceDetector initialized: {self.model_name} (conf={self.conf_threshold})")
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar RetinaFace: {e}")
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detecta faces na imagem.
        
        Args:
            image: Imagem numpy (RGB ou BGR)
        
        Returns:
            Lista de dicionários com informações das faces detectadas:
            [
                {
                    'box': [x1, y1, x2, y2],
                    'landmarks': np.array shape (5, 2),
                    'score': float
                },
                ...
            ]
        """
        if self.detector is None:
            self._init_detector()
        
        # uniface.RetinaFace.detect() retorna uma lista de objetos Face
        faces_raw = self.detector.detect(image)
        
        # Converter para formato padronizado
        faces = []
        
        if faces_raw is None:
            return faces
        
        # Iterar sobre as faces detectadas
        for face in faces_raw:
            face_dict = self._parse_face_object(face)
            if face_dict is not None:
                faces.append(face_dict)
        
        # Ordenar por score (maior primeiro)
        faces.sort(key=lambda x: x['score'], reverse=True)
        
        return faces
    
    def _parse_face_object(self, face) -> Optional[dict]:
        """
        Parseia um objeto Face do uniface para dict padronizado.
        
        O uniface pode retornar diferentes formatos dependendo da versão.
        Esta função tenta múltiplas formas de extrair os dados.
        """
        face_dict = {}
        
        try:
            # ===== BOUNDING BOX =====
            bbox = None
            
            # Tentar atributo bbox
            if hasattr(face, 'bbox'):
                bbox = face.bbox
            elif hasattr(face, 'box'):
                bbox = face.box
            
            if bbox is not None:
                if hasattr(bbox, 'tolist'):
                    face_dict['box'] = bbox.tolist()
                elif hasattr(bbox, '__iter__'):
                    face_dict['box'] = list(bbox)
                else:
                    return None
            else:
                return None
            
            # ===== LANDMARKS =====
            landmarks = None
            
            # Tentar atributo landmarks ou kps
            if hasattr(face, 'landmarks'):
                landmarks = face.landmarks
            elif hasattr(face, 'kps'):
                landmarks = face.kps
            elif hasattr(face, 'keypoints'):
                landmarks = face.keypoints
            
            if landmarks is not None:
                landmarks = np.array(landmarks)
                # Garantir shape (5, 2)
                if landmarks.size == 10:
                    landmarks = landmarks.reshape(5, 2)
                elif landmarks.shape == (5, 2):
                    pass
                else:
                    # Tentar usar os primeiros 10 valores
                    landmarks = landmarks.flatten()[:10].reshape(5, 2)
                
                face_dict['landmarks'] = landmarks.astype(np.float32)
            else:
                return None
            
            # ===== SCORE =====
            score = 1.0
            
            if hasattr(face, 'det_score'):
                score = float(face.det_score)
            elif hasattr(face, 'score'):
                score = float(face.score)
            elif hasattr(face, 'confidence'):
                score = float(face.confidence)
            
            face_dict['score'] = score
            
            return face_dict
            
        except Exception as e:
            print(f"Warning: Failed to parse face object: {e}")
            return None
    
    def detect_largest_face(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Detecta e retorna a maior face na imagem.
        
        Args:
            image: Imagem numpy
        
        Returns:
            Tuple (box, landmarks) ou (None, None) se nenhuma face
        """
        faces = self.detect(image)
        
        if not faces:
            return None, None
        
        # Encontrar face com maior área
        largest_face = None
        largest_area = 0
        
        for face in faces:
            box = face['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > largest_area:
                largest_area = area
                largest_face = face
        
        if largest_face is None:
            return None, None
        
        return largest_face['box'], largest_face['landmarks']
    
    def detect_most_confident_face(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Detecta e retorna a face com maior score de confiança.
        
        Args:
            image: Imagem numpy
        
        Returns:
            Tuple (box, landmarks) ou (None, None) se nenhuma face
        """
        faces = self.detect(image)
        
        if not faces:
            return None, None
        
        # Faces já estão ordenadas por score
        best_face = faces[0]
        
        return best_face['box'], best_face['landmarks']
    
    def detect_and_align(
        self,
        image: np.ndarray,
        image_size: int = 112,
        select_largest: bool = True
    ) -> np.ndarray:
        """
        Detecta face e retorna imagem alinhada.
        
        Args:
            image: Imagem numpy (RGB ou BGR)
            image_size: Tamanho da saída
            select_largest: Se True, seleciona maior face; senão, mais confiante
        
        Returns:
            Imagem alinhada (image_size, image_size, 3)
        
        Raises:
            NoFaceDetectedError: Se nenhuma face for detectada
        """
        if select_largest:
            box, landmarks = self.detect_largest_face(image)
        else:
            box, landmarks = self.detect_most_confident_face(image)
        
        if landmarks is None:
            raise NoFaceDetectedError("Nenhuma face detectada na imagem")
        
        return align_face(image, landmarks, image_size)
    
    def get_detection_info(self, image: np.ndarray) -> dict:
        """
        Retorna informações detalhadas sobre as detecções.
        
        Args:
            image: Imagem numpy
        
        Returns:
            Dict com informações das detecções
        """
        faces = self.detect(image)
        
        return {
            'num_faces': len(faces),
            'faces': faces,
            'image_size': image.shape[:2]
        }


# ===========================================
# Funções de Conveniência (API Funcional)
# ===========================================
_detector_instance = None


def _get_detector(conf_threshold: float = 0.35) -> FaceDetector:
    """Obtém instância singleton do detector."""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = FaceDetector(conf_threshold=conf_threshold)
    
    return _detector_instance


def detect_and_align_face(
    image: Union[np.ndarray, 'PIL.Image.Image', str, Path],
    image_size: int = 112,
    conf_threshold: float = 0.35,
    select_largest: bool = True,
    return_info: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Detecta e alinha face em uma imagem.
    
    Esta é a função principal de conveniência para uso simplificado.
    
    Args:
        image: Imagem (numpy array, PIL Image, ou caminho)
        image_size: Tamanho da saída (padrão: 112)
        conf_threshold: Limiar de confiança do detector
        select_largest: Se True, seleciona maior face
        return_info: Se True, retorna também informações da detecção
    
    Returns:
        Se return_info=False: Array numpy (image_size, image_size, 3) RGB
        Se return_info=True: Tuple (aligned_face, detection_info)
    
    Raises:
        NoFaceDetectedError: Se nenhuma face for detectada
    
    Exemplo:
        # Uso simples
        aligned = detect_and_align_face("foto.jpg")
        
        # Com informações
        aligned, info = detect_and_align_face("foto.jpg", return_info=True)
        print(f"Faces detectadas: {info['num_faces']}")
    """
    # Converter para numpy se necessário
    image_np = _load_image_as_numpy(image)
    
    # Obter detector
    detector = _get_detector(conf_threshold)
    
    # Detectar e alinhar
    aligned = detector.detect_and_align(
        image_np,
        image_size=image_size,
        select_largest=select_largest
    )
    
    if return_info:
        info = detector.get_detection_info(image_np)
        return aligned, info
    
    return aligned


def detect_faces(
    image: Union[np.ndarray, 'PIL.Image.Image', str, Path],
    conf_threshold: float = 0.35
) -> List[dict]:
    """
    Detecta todas as faces em uma imagem.
    
    Args:
        image: Imagem (numpy array, PIL Image, ou caminho)
        conf_threshold: Limiar de confiança
    
    Returns:
        Lista de dicionários com informações das faces
    """
    image_np = _load_image_as_numpy(image)
    detector = _get_detector(conf_threshold)
    return detector.detect(image_np)


def detect_and_align_face_pil(
    image: Union[np.ndarray, 'PIL.Image.Image', str, Path],
    image_size: int = 112,
    conf_threshold: float = 0.35,
    select_largest: bool = True
) -> 'PIL.Image.Image':
    """
    Detecta e alinha face, retornando PIL Image.
    
    Args:
        image: Imagem de entrada
        image_size: Tamanho da saída
        conf_threshold: Limiar de confiança
        select_largest: Selecionar maior face
    
    Returns:
        PIL Image com face alinhada
    
    Raises:
        NoFaceDetectedError: Se nenhuma face detectada
    """
    from PIL import Image as PILImage
    
    aligned_np = detect_and_align_face(
        image,
        image_size=image_size,
        conf_threshold=conf_threshold,
        select_largest=select_largest
    )
    
    return PILImage.fromarray(aligned_np)


# ===========================================
# Funções Auxiliares
# ===========================================
def _load_image_as_numpy(
    image: Union[np.ndarray, 'PIL.Image.Image', str, Path]
) -> np.ndarray:
    """
    Carrega imagem de várias fontes e retorna numpy array RGB.
    
    Args:
        image: numpy array, PIL Image, ou caminho para arquivo
    
    Returns:
        Numpy array (H, W, 3) em RGB
    """
    # Se já é numpy array
    if isinstance(image, np.ndarray):
        # Assumir que pode ser BGR (OpenCV) ou RGB
        # Manter como está - o detector lida com ambos
        return image
    
    # Se é string ou Path (caminho para arquivo)
    if isinstance(image, (str, Path)):
        # Usar OpenCV para carregar (retorna BGR)
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            raise FileNotFoundError(f"Não foi possível carregar imagem: {image}")
        # Converter BGR para RGB
        return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Se é PIL Image
    try:
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
    except ImportError:
        pass
    
    raise TypeError(f"Tipo de imagem não suportado: {type(image)}")


def is_face_detection_available() -> bool:
    """
    Verifica se a detecção facial está disponível.
    
    Returns:
        True se uniface e skimage estão instalados
    """
    return UNIFACE_AVAILABLE and SKIMAGE_AVAILABLE


def draw_detection(
    image: np.ndarray,
    faces: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Desenha bounding boxes e landmarks na imagem.
    
    Args:
        image: Imagem numpy
        faces: Lista de detecções do método detect()
        color: Cor do desenho (BGR)
        thickness: Espessura das linhas
    
    Returns:
        Imagem com desenhos
    """
    output = image.copy()
    
    for face in faces:
        box = face['box']
        landmarks = face['landmarks']
        score = face.get('score', 0)
        
        # Desenhar bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Desenhar score
        cv2.putText(
            output,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness
        )
        
        # Desenhar landmarks
        for point in landmarks:
            x, y = map(int, point)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
    
    return output


# ===========================================
# Script de teste
# ===========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test face detection module")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--output", type=str, default=None, help="Output path for aligned face")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--draw", action="store_true", help="Draw detections on image")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Face Detection Module - Test")
    print("=" * 60)
    
    # Verificar disponibilidade
    print(f"\nuniFace available: {UNIFACE_AVAILABLE}")
    print(f"scikit-image available: {SKIMAGE_AVAILABLE}")
    
    if not is_face_detection_available():
        print("\n✗ Face detection not available!")
        print("Install requirements: pip install uniface scikit-image")
        exit(1)
    
    # Carregar imagem
    print(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"✗ Failed to load image: {args.image}")
        exit(1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image.shape}")
    
    # Detectar faces
    print(f"\nDetecting faces (conf={args.conf})...")
    try:
        aligned, info = detect_and_align_face(
            image_rgb,
            conf_threshold=args.conf,
            return_info=True
        )
        
        print(f"✓ Face detected!")
        print(f"  Num faces: {info['num_faces']}")
        print(f"  Aligned shape: {aligned.shape}")
        
        if info['faces']:
            best_face = info['faces'][0]
            print(f"  Best score: {best_face['score']:.3f}")
            print(f"  Box: {best_face['box']}")
        
        # Salvar resultado
        if args.output:
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, aligned_bgr)
            print(f"\n✓ Aligned face saved to: {args.output}")
        
        # Desenhar detecções
        if args.draw:
            drawn = draw_detection(image, info['faces'])
            draw_path = args.output.replace('.', '_drawn.') if args.output else 'detection_result.jpg'
            cv2.imwrite(draw_path, drawn)
            print(f"✓ Detection visualization saved to: {draw_path}")
        
    except NoFaceDetectedError as e:
        print(f"✗ No face detected: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)