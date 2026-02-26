"""
Detecção facial e alinhamento usando SCRFD (InsightFace) + face_align.norm_crop.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List

# ---------------------------------------------------------------------------
# Importação condicional do InsightFace (SCRFD + face_align)
# ---------------------------------------------------------------------------
try:
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    SCRFD = None
    face_align = None


# ===========================================================================
# Exceções Customizadas  (mantidas idênticas para compatibilidade)
# ===========================================================================
class NoFaceDetectedError(Exception):
    """Levantada quando nenhuma face é detectada na imagem."""
    pass


class MultipleFacesDetectedError(Exception):
    """Levantada quando múltiplas faces são detectadas (se não permitido)."""
    pass


# ===========================================================================
# Constantes
# ===========================================================================
# Caminhos padrão onde o SCRFD pode estar presente localmente
_SCRFD_CANDIDATES = [
    Path.home() / ".insightface/models/scrfd_10g_bnkps.onnx",
    Path.home() / ".insightface/models/buffalo_l/det_10g.onnx",
    Path("models/scrfd_10g_bnkps.onnx"),
    Path("scrfd_10g_bnkps.onnx"),
]

# Mantido para compatibilidade com código que importe REFERENCE_ALIGNMENT
REFERENCE_ALIGNMENT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ===========================================================================
# Helpers internos
# ===========================================================================
def _find_scrfd_model() -> Optional[Path]:
    """Procura o arquivo .onnx do SCRFD nos caminhos padrão."""
    for candidate in _SCRFD_CANDIDATES:
        if Path(candidate).exists():
            return Path(candidate)
    return None


def _load_image_as_numpy(
    image: Union[np.ndarray, "PIL.Image.Image", str, Path]
) -> np.ndarray:
    """
    Converte qualquer fonte de imagem para numpy BGR uint8.
    Mantida para compatibilidade com detect_and_align_face().
    """
    if isinstance(image, np.ndarray):
        return image

    # PIL Image
    try:
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except ImportError:
        pass

    # Caminho
    path = str(image)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {path}")
    return img


def crop_from_bbox(
    image: np.ndarray,
    bbox: List[float],
    image_size: int = 112,
    expand: float = 1.40,
) -> np.ndarray:
    """
    Fallback: recorta e redimensiona a região da face a partir da bounding box,
    sem alinhamento de landmarks.  Equivalente ao crop_from_bbox do cropping_vgg.py.

    Args:
        image:      Imagem BGR numpy.
        bbox:       [x1, y1, x2, y2] em pixels.
        image_size: Tamanho do crop de saída (quadrado).
        expand:     Fator de expansão da bbox para incluir contexto.

    Returns:
        Crop redimensionado para (image_size, image_size, 3) BGR.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Centro e dimensões
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * expand, (y2 - y1) * expand

    # Expandir e clipar
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2))
    ny2 = min(h, int(cy + bh / 2))

    crop = image[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = image  # fallback extremo: imagem inteira

    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


# ===========================================================================
# Funções públicas de alinhamento (mantidas para compatibilidade)
# ===========================================================================
def get_reference_alignment(image_size: int = 112) -> np.ndarray:
    """Retorna pontos de referência ArcFace ajustados ao tamanho."""
    if image_size % 112 == 0:
        ratio, diff_x = float(image_size) / 112.0, 0.0
    elif image_size % 128 == 0:
        ratio, diff_x = float(image_size) / 128.0, 8.0 * (float(image_size) / 128.0)
    else:
        ratio, diff_x = float(image_size) / 112.0, 0.0

    ref = REFERENCE_ALIGNMENT.copy() * ratio
    ref[:, 0] += diff_x
    return ref


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Mantida para compatibilidade.
    Estima matriz de transformação via SimilarityTransform (skimage).
    """
    try:
        from skimage.transform import SimilarityTransform
    except ImportError:
        raise RuntimeError("scikit-image necessário: pip install scikit-image")

    assert landmark.shape == (5, 2), f"Esperado (5,2), recebido {landmark.shape}"
    reference = get_reference_alignment(image_size)
    transform = SimilarityTransform()
    transform.estimate(landmark, reference)
    return transform.params[0:2, :]


def align_face(
    image: np.ndarray,
    landmark: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    """
    Mantida para compatibilidade.
    Alinha face via SimilarityTransform+warpAffine (comportamento anterior).
    O novo pipeline usa norm_crop internamente via FaceDetector.
    """
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)


# ===========================================================================
# Classe FaceDetector  (API pública mantida idêntica)
# ===========================================================================
class FaceDetector:
    """
    Detector facial usando SCRFD (InsightFace) com alinhamento via norm_crop.

    Mantém a mesma interface pública da versão anterior (RetinaFace/uniface)
    para compatibilidade com todo o código que importa FaceDetector.

    Singleton: o modelo SCRFD é carregado uma única vez.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "scrfd_10g_bnkps",   # mantido como parâmetro p/ compatibilidade
        conf_threshold: float = 0.35,
        image_size: int = 112,
        expand_fallback: float = 1.40,
    ):
        """
        Args:
            model_name:       Ignorado (mantido para compatibilidade de assinatura).
            conf_threshold:   Limiar de confiança do SCRFD.
            image_size:       Tamanho de saída do crop alinhado.
            expand_fallback:  Fator de expansão da bbox no fallback sem landmarks.
        """
        if self._initialized:
            return

        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface é necessário: pip install insightface\n"
                "Baixe o modelo SCRFD:\n"
                "  wget -O ~/.insightface/models/scrfd_10g_bnkps.onnx \\\n"
                "    https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
            )

        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.expand_fallback = expand_fallback

        # Localizar e carregar SCRFD
        scrfd_path = _find_scrfd_model()
        if scrfd_path is None:
            raise FileNotFoundError(
                "Modelo SCRFD não encontrado. Baixe com:\n"
                "  wget -O ~/.insightface/models/scrfd_10g_bnkps.onnx \\\n"
                "    https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
            )

        self.detector = SCRFD(model_file=str(scrfd_path))
        # det_thresh e input_size são passados no prepare(), não no detect()
        self.detector.prepare(ctx_id=-1, det_thresh=conf_threshold, input_size=(640, 640))

        self.__class__._initialized = True
        print(f"✓ FaceDetector initialized: SCRFD ({scrfd_path.name}, conf={conf_threshold})")

    # ------------------------------------------------------------------
    # Método de detecção bruta (retorna lista de dicts como antes)
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detecta faces e retorna lista de dicts com 'box', 'landmarks' e 'score'.

        Args:
            image: numpy BGR.

        Returns:
            Lista ordenada por score (maior primeiro).
            Cada item: {'box': [x1,y1,x2,y2], 'landmarks': np.ndarray(5,2), 'score': float}
        """
        # SCRFD espera BGR; retorna bboxes (N,5) e kps (N,5,2)
        # det_thresh e input_size são configurados no prepare(), não aqui
        bboxes, kps = self.detector.detect(image)

        if bboxes is None or len(bboxes) == 0:
            return []

        faces = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            landmarks = kps[i] if kps is not None and i < len(kps) else None
            faces.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "landmarks": landmarks,   # np.ndarray (5,2) ou None
                "score": float(score),
            })

        # Ordenar por score decrescente
        faces.sort(key=lambda f: f["score"], reverse=True)
        return faces

    # ------------------------------------------------------------------
    # Seleção de face
    # ------------------------------------------------------------------
    def detect_largest_face(
        self, image: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """Retorna (box, landmarks) da face com maior área."""
        faces = self.detect(image)
        if not faces:
            return None, None

        largest = max(
            faces,
            key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
        )
        return largest["box"], largest["landmarks"]

    def detect_most_confident_face(
        self, image: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """Retorna (box, landmarks) da face com maior score de confiança."""
        faces = self.detect(image)
        if not faces:
            return None, None
        best = faces[0]  # já ordenado por score
        return best["box"], best["landmarks"]

    # ------------------------------------------------------------------
    # Alinhamento principal
    # ------------------------------------------------------------------
    def detect_and_align(
        self,
        image: np.ndarray,
        image_size: int = 112,
        select_largest: bool = True,
    ) -> np.ndarray:
        """
        Detecta face e retorna imagem alinhada.

        Usa face_align.norm_crop (InsightFace) quando landmarks estão disponíveis.
        Usa crop_from_bbox como fallback quando landmarks não são detectados.

        Args:
            image:          numpy BGR.
            image_size:     Tamanho de saída do crop.
            select_largest: Se True, seleciona maior face; senão, mais confiante.

        Returns:
            Crop alinhado (image_size, image_size, 3) BGR.

        Raises:
            NoFaceDetectedError: Quando nenhuma face é detectada.
        """
        if select_largest:
            box, landmarks = self.detect_largest_face(image)
        else:
            box, landmarks = self.detect_most_confident_face(image)

        if box is None:
            raise NoFaceDetectedError("Nenhuma face detectada na imagem.")

        if landmarks is not None:
            # Alinhamento completo com norm_crop (InsightFace)
            aligned = face_align.norm_crop(image, landmark=landmarks, image_size=image_size)
            return aligned
        else:
            # Fallback: crop simples expandido a partir da bbox
            return crop_from_bbox(image, box, image_size=image_size, expand=self.expand_fallback)

    def get_detection_info(self, image: np.ndarray) -> dict:
        """Retorna informações detalhadas das detecções."""
        faces = self.detect(image)
        return {
            "num_faces": len(faces),
            "faces": faces,
            "image_size": image.shape[:2],
        }


# ===========================================================================
# API Funcional (compatibilidade total com código existente)
# ===========================================================================
_detector_instance: Optional[FaceDetector] = None


def _get_detector(conf_threshold: float = 0.35) -> FaceDetector:
    """Retorna instância singleton do FaceDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(conf_threshold=conf_threshold)
    return _detector_instance


def detect_and_align_face(
    image: Union[np.ndarray, "PIL.Image.Image", str, Path],
    image_size: int = 112,
    conf_threshold: float = 0.35,
    select_largest: bool = True,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Detecta e alinha face em uma imagem.  API pública mantida idêntica.

    Args:
        image:          numpy, PIL Image ou caminho.
        image_size:     Tamanho de saída (padrão: 112).
        conf_threshold: Limiar de confiança do detector.
        select_largest: Se True, seleciona maior face.
        return_info:    Se True, retorna também informações da detecção.

    Returns:
        Array numpy (image_size, image_size, 3) BGR  — ou  (aligned, info).

    Raises:
        NoFaceDetectedError: Se nenhuma face for detectada.
    """
    image_np = _load_image_as_numpy(image)
    detector = _get_detector(conf_threshold)

    aligned = detector.detect_and_align(
        image_np, image_size=image_size, select_largest=select_largest
    )

    if return_info:
        info = detector.get_detection_info(image_np)
        return aligned, info

    return aligned


def detect_faces(
    image: Union[np.ndarray, "PIL.Image.Image", str, Path],
    conf_threshold: float = 0.35,
) -> List[dict]:
    """Detecta todas as faces em uma imagem."""
    image_np = _load_image_as_numpy(image)
    return _get_detector(conf_threshold).detect(image_np)


def is_face_detection_available() -> bool:
    """Retorna True se insightface está instalado."""
    return INSIGHTFACE_AVAILABLE


def draw_detection(
    image: np.ndarray,
    faces: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Desenha bounding boxes e landmarks na imagem."""
    output = image.copy()
    for face in faces:
        box = face["box"]
        landmarks = face.get("landmarks")
        score = face.get("score", 0)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            output, f"{score:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness,
        )
        if landmarks is not None:
            for point in landmarks:
                cx, cy = map(int, point)
                cv2.circle(output, (cx, cy), 2, (0, 0, 255), -1)

    return output