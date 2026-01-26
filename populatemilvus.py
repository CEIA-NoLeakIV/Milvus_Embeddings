import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
from PIL import Image

from app.config import Config
from app.milvus_client import MilvusClient
from models import ModelFactory

# IMPORTANTE: Usar preprocessing centralizado
from preprocessing import extract_embedding_standardized

# Importar exceção de detecção facial
try:
    from utils.face_detection import NoFaceDetectedError, is_face_detection_available
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    NoFaceDetectedError = Exception


# ===========================================
# Configurações
# ===========================================
LFW_DIR = ROOT_DIR / "lfw"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Popular Milvus com embeddings do LFW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python populatemilvus.py
    python populatemilvus.py --model cosface_resnet50
    python populatemilvus.py --limit 100 --recreate
    python populatemilvus.py --no-face-detection
    python populatemilvus.py --skip-no-face --face-conf 0.6
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv3_large",
        choices=Config.AVAILABLE_MODELS,
        help=f"Modelo para extração (default: mobilenetv3_large)"
    )
    
    parser.add_argument(
        "--lfw-dir",
        type=str,
        default=str(LFW_DIR),
        help=f"Diretório do dataset LFW (default: {LFW_DIR})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Tamanho do lote para inserção no Milvus (default: 50)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar número de imagens processadas (default: todas)"
    )
    
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recriar collection (apaga dados existentes)"
    )
    
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Desabilitar TTA (usar 512 dims ao invés de 1024)"
    )
    
    # ===========================================
    # Novos argumentos para detecção facial
    # ===========================================
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help="Desabilitar detecção facial (usa imagem original)"
    )
    
    parser.add_argument(
        "--skip-no-face",
        action="store_true",
        help="Pular imagens onde nenhuma face é detectada (ao invés de erro)"
    )
    
    parser.add_argument(
        "--face-conf",
        type=float,
        default=0.35,
        help="Limiar de confiança para detecção facial (default: 0.5)"
    )
    
    parser.add_argument(
        "--select-largest",
        action="store_true",
        default=True,
        help="Selecionar a maior face quando múltiplas são detectadas (default: True)"
    )
    
    parser.add_argument(
        "--save-no-face-list",
        type=str,
        default=None,
        help="Salvar lista de imagens sem face detectada em arquivo"
    )
    
    return parser.parse_args()


def get_image_files(directory: str) -> list:
    """Lista arquivos de imagem recursivamente."""
    image_files = []
    path_obj = Path(directory)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(path_obj.rglob(f"*{ext}"))
        image_files.extend(path_obj.rglob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def extract_person_id(image_path: Path, base_dir: Path) -> str:
    """Extrai o ID da pessoa do caminho da imagem."""
    try:
        relative = image_path.relative_to(base_dir)
        return relative.parts[0] if relative.parts else image_path.stem
    except ValueError:
        return image_path.parent.name


def main():
    args = parse_args()
    
    # Determinar uso de TTA
    use_tta = not args.no_tta
    embedding_dim = 1024 if use_tta else 512
    
    # Determinar uso de detecção facial
    use_face_detection = not args.no_face_detection
    
    # Verificar disponibilidade
    if use_face_detection and not FACE_DETECTION_AVAILABLE:
        print("⚠️  Face detection requested but uniface not installed!")
        print("   Install with: pip install uniface")
        print("   Continuing without face detection...")
        use_face_detection = False
    
    print("=" * 60)
    print("  POPULAR MILVUS COM EMBEDDINGS")
    print("=" * 60)
    print(f"  Modelo:           {args.model}")
    print(f"  TTA:              {'ON (1024 dims)' if use_tta else 'OFF (512 dims)'}")
    print(f"  Embedding dim:    {embedding_dim}")
    print(f"  LFW dir:          {args.lfw_dir}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Limit:            {args.limit or 'Todas'}")
    print(f"  Recreate:         {args.recreate}")
    print()
    print("  --- Face Detection ---")
    print(f"  Enabled:          {use_face_detection}")
    if use_face_detection:
        print(f"  Conf threshold:   {args.face_conf}")
        print(f"  Select largest:   {args.select_largest}")
        print(f"  Skip no-face:     {args.skip_no_face}")
    print("=" * 60)
    
    # IMPORTANTE: Atualizar Config ANTES de criar qualquer cliente
    Config.USE_TTA = use_tta
    Config.EMBEDDING_DIM = embedding_dim
    Config.USE_FACE_DETECTION = use_face_detection
    Config.FACE_DETECTION_CONF_THRESHOLD = args.face_conf
    Config.FACE_DETECTION_SELECT_LARGEST = args.select_largest
    
    # Carregar modelo com a configuração correta de TTA
    print("\n[1/4] Carregando modelo...")
    model = ModelFactory.create(args.model, use_tta=use_tta)
    print(f"✓ Modelo carregado: {model}")
    
    # Conectar ao Milvus
    print("\n[2/4] Conectando ao Milvus...")
    milvus = MilvusClient()
    
    if args.recreate:
        print("Recriando collection...")
        milvus.recreate_collection()
    
    # Listar imagens
    print("\n[3/4] Listando imagens...")
    image_files = get_image_files(args.lfw_dir)
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"✓ {len(image_files)} imagens encontradas")
    
    # Extrair e inserir embeddings
    print("\n[4/4] Extraindo embeddings e inserindo no Milvus...")
    if use_face_detection:
        print("      (Com detecção facial + alinhamento)")
    else:
        print("      (Sem detecção facial - usando imagem original)")
    
    base_dir = Path(args.lfw_dir)
    batch_embeddings = []
    batch_person_ids = []
    batch_image_paths = []
    
    total_inserted = 0
    errors = 0
    no_face_count = 0
    no_face_images = []  # Lista de imagens sem face
    
    for img_path in tqdm(image_files, desc="Processando"):
        try:
            # IMPORTANTE: Usar preprocessing centralizado com detecção facial
            embedding = extract_embedding_standardized(
                model,
                file_path=str(img_path),
                use_tta=use_tta,
                use_face_detection=use_face_detection,
                conf_threshold=args.face_conf,
                select_largest=args.select_largest
            )
            
            # Extrair person_id
            person_id = extract_person_id(img_path, base_dir)
            
            # Adicionar ao batch
            batch_embeddings.append(embedding)
            batch_person_ids.append(person_id)
            batch_image_paths.append(str(img_path))
            
            # Inserir quando batch estiver cheio
            if len(batch_embeddings) >= args.batch_size:
                milvus.insert(
                    embeddings=batch_embeddings,
                    person_ids=batch_person_ids,
                    image_paths=batch_image_paths
                )
                total_inserted += len(batch_embeddings)
                batch_embeddings = []
                batch_person_ids = []
                batch_image_paths = []
                
        except NoFaceDetectedError as e:
            no_face_count += 1
            no_face_images.append(str(img_path))
            
            if args.skip_no_face:
                # Apenas registrar e continuar
                tqdm.write(f"⚠ Sem face: {img_path.name}")
            else:
                # Tratar como erro
                errors += 1
                tqdm.write(f"✗ Sem face: {img_path.name}")
                
        except Exception as e:
            errors += 1
            tqdm.write(f"✗ Erro em {img_path.name}: {e}")
    
    # Inserir batch restante
    if batch_embeddings:
        milvus.insert(
            embeddings=batch_embeddings,
            person_ids=batch_person_ids,
            image_paths=batch_image_paths
        )
        total_inserted += len(batch_embeddings)
    
    # Salvar lista de imagens sem face (se solicitado)
    if args.save_no_face_list and no_face_images:
        no_face_file = Path(args.save_no_face_list)
        no_face_file.parent.mkdir(exist_ok=True, parents=True)
        with open(no_face_file, 'w') as f:
            for img_path in no_face_images:
                f.write(f"{img_path}\n")
        print(f"\n✓ Lista de imagens sem face salva em: {no_face_file}")
    
    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  Total processado:   {len(image_files)}")
    print(f"  Total inserido:     {total_inserted}")
    print(f"  Erros:              {errors}")
    if use_face_detection:
        print(f"  Sem face detectada: {no_face_count}")
        if args.skip_no_face:
            print(f"    (puladas, não contam como erro)")
    print()
    print(f"  Embedding dim:      {embedding_dim}")
    print(f"  TTA:                {'ON' if use_tta else 'OFF'}")
    print(f"  Face detection:     {'ON' if use_face_detection else 'OFF'}")
    
    # Stats da collection
    stats = milvus.get_collection_stats()
    print(f"  Collection size:    {stats['row_count']}")
    print("=" * 60)
    
    # Salvar configuração usada para referência
    config_file = ROOT_DIR / "data" / "populate_config.txt"
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"use_tta: {use_tta}\n")
        f.write(f"embedding_dim: {embedding_dim}\n")
        f.write(f"use_face_detection: {use_face_detection}\n")
        f.write(f"face_conf_threshold: {args.face_conf}\n")
        f.write(f"select_largest: {args.select_largest}\n")
        f.write(f"total_processed: {len(image_files)}\n")
        f.write(f"total_inserted: {total_inserted}\n")
        f.write(f"no_face_count: {no_face_count}\n")
        f.write(f"errors: {errors}\n")
    print(f"\n✓ Configuração salva em: {config_file}")
    
    # Retornar código de saída apropriado
    if errors > 0 and not args.skip_no_face:
        print(f"\n⚠️  {errors} erros encontrados. Verifique os logs acima.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())