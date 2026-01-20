import io
from pathlib import Path
from typing import Union, Optional, BinaryIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ===========================================
# Configurações de Pré-processamento
# ===========================================
IMAGE_SIZE = (112, 112)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
INTERPOLATION = Image.BILINEAR  # Padronizar interpolação


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
# Função de Pré-processamento Principal
# ===========================================
def preprocess_image(
    file_path: Optional[Union[str, Path]] = None,
    pil_image: Optional[Image.Image] = None,
    file_bytes: Optional[bytes] = None,
    file_stream: Optional[BinaryIO] = None,
    return_flipped: bool = False,
    device: torch.device = None
) -> Union[torch.Tensor, tuple]:
    """
    Pré-processa imagem de qualquer fonte para tensor pronto para o modelo.
    
    Esta é a função PRINCIPAL que deve ser usada em TODO lugar:
    - populatemilvus.py
    - API Flask
    - Streamlit
    - Scripts de teste
    
    Args:
        file_path: Caminho para arquivo
        pil_image: Imagem PIL
        file_bytes: Bytes da imagem
        file_stream: Stream de arquivo
        return_flipped: Se True, retorna também versão flipped (para TTA)
        device: Device para mover o tensor (cuda/cpu)
    
    Returns:
        Se return_flipped=False: tensor (1, 3, 112, 112)
        Se return_flipped=True: tuple (tensor_original, tensor_flipped)
    """
    # 1. Carregar imagem de forma padronizada
    image = load_image_standardized(
        file_path=file_path,
        pil_image=pil_image,
        file_bytes=file_bytes,
        file_stream=file_stream
    )
    
    # 2. Aplicar transform
    tensor = _transform(image)
    
    # 3. Adicionar dimensão de batch
    tensor = tensor.unsqueeze(0)
    
    # 4. Mover para device se especificado
    if device is not None:
        tensor = tensor.to(device)
    
    # 5. Se precisa da versão flipped (para TTA)
    if return_flipped:
        tensor_flip = _transform_flip(image).unsqueeze(0)
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
    use_tta: bool = True
) -> np.ndarray:
    """
    Extrai embedding de forma totalmente padronizada.
    
    Esta função garante que o pré-processamento seja IDÊNTICO
    independente da fonte da imagem.
    
    Args:
        model: Modelo de face recognition (deve ter .model e .device)
        file_path: Caminho para arquivo
        pil_image: Imagem PIL
        file_bytes: Bytes da imagem
        file_stream: Stream de arquivo
        use_tta: Se True, usa Test-Time Augmentation (concatena original + flip)
    
    Returns:
        Embedding como numpy array (512 dims sem TTA, 1024 com TTA)
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
            device=device
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
            device=device
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
    
    def __init__(self, base_model, use_tta: bool = None):
        """
        Args:
            base_model: Modelo base (MobileNetModel, CosFaceModel, etc)
            use_tta: Sobrescreve configuração de TTA do modelo base
        """
        self.base_model = base_model
        self.model = base_model.model
        self.device = base_model.device
        self.model_name = base_model.model_name
        
        # Usar TTA do modelo base se não especificado
        self.use_tta = use_tta if use_tta is not None else base_model.use_tta
        self.output_dim = 1024 if self.use_tta else 512
    
    def extract_embedding_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """Extrai embedding de um arquivo."""
        return extract_embedding_standardized(
            self,
            file_path=image_path,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        """Extrai embedding de uma imagem PIL."""
        return extract_embedding_standardized(
            self,
            pil_image=image,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """Extrai embedding de bytes."""
        return extract_embedding_standardized(
            self,
            file_bytes=file_bytes,
            use_tta=self.use_tta
        )
    
    def extract_embedding_from_stream(self, file_stream: BinaryIO) -> np.ndarray:
        """Extrai embedding de um stream de arquivo."""
        return extract_embedding_standardized(
            self,
            file_stream=file_stream,
            use_tta=self.use_tta
        )
    
    def __repr__(self) -> str:
        return (
            f"StandardizedModelWrapper("
            f"model={self.model_name}, "
            f"use_tta={self.use_tta}, "
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
    # Carregar de diferentes formas
    tensor_from_path = preprocess_image(file_path=image_path)
    
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    tensor_from_bytes = preprocess_image(file_bytes=file_bytes)
    
    with open(image_path, 'rb') as f:
        tensor_from_stream = preprocess_image(file_stream=f)
    
    pil_image = Image.open(image_path).convert('RGB')
    tensor_from_pil = preprocess_image(pil_image=pil_image)
    
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    
    if args.verify:
        print("Verificando consistência do pré-processamento...")
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
    else:
        # Demo de uso
        print("Demo de pré-processamento padronizado")
        print("=" * 50)
        
        # Carregar de diferentes formas
        tensor = preprocess_image(file_path=args.image)
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        
        # Com TTA
        tensor_orig, tensor_flip = preprocess_image(file_path=args.image, return_flipped=True)
        print(f"\nCom TTA:")
        print(f"  Original: {tensor_orig.shape}")
        print(f"  Flipped: {tensor_flip.shape}")