#!/usr/bin/env python3
"""
Script para popular o Milvus com embeddings do dataset LFW
==========================================================

Este script:
1. Carrega o modelo MobileNetV3 Large com TTA
2. Percorre todas as imagens do dataset LFW
3. Extrai os embeddings de cada imagem (1024 dims com TTA)
4. Insere no Milvus Lite (salvo em data/milvus_face.db)

Uso:
    python populatemilvus.py
    python populatemilvus.py --model cosface_resnet50
    python populatemilvus.py --batch-size 64 --limit 1000
    python populatemilvus.py --no-tta  # Para usar 512 dims sem TTA
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
from PIL import Image

from app.config import Config
from app.milvus_client import MilvusClient
from models import ModelFactory


# ===========================================
# Configurações
# ===========================================
LFW_DIR = ROOT_DIR / "lfw"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Popular Milvus com embeddings do LFW",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    """
    Extrai o ID da pessoa do caminho da imagem.
    Assume estrutura: base_dir/person_name/image.jpg
    """
    try:
        relative = image_path.relative_to(base_dir)
        return relative.parts[0] if relative.parts else image_path.stem
    except ValueError:
        return image_path.parent.name


def main():
    args = parse_args()
    
    # Configurações
    use_tta = not args.no_tta
    embedding_dim = 1024 if use_tta else 512
    
    print("=" * 60)
    print("  POPULAR MILVUS COM EMBEDDINGS")
    print("=" * 60)
    print(f"  Modelo:        {args.model}")
    print(f"  TTA:           {'ON (1024 dims)' if use_tta else 'OFF (512 dims)'}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  LFW dir:       {args.lfw_dir}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Limit:         {args.limit or 'Todas'}")
    print(f"  Recreate:      {args.recreate}")
    print("=" * 60)
    
    # Carregar modelo com TTA
    print("\n[1/4] Carregando modelo...")
    model = ModelFactory.create(args.model, use_tta=use_tta)
    print(f"✓ Modelo carregado: {model}")
    
    # Conectar ao Milvus
    print("\n[2/4] Conectando ao Milvus...")
    
    # Se usar TTA, precisa atualizar Config antes de criar o cliente
    if use_tta:
        Config.EMBEDDING_DIM = 1024
    else:
        Config.EMBEDDING_DIM = 512
    
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
    
    base_dir = Path(args.lfw_dir)
    batch_embeddings = []
    batch_person_ids = []
    batch_image_paths = []
    
    total_inserted = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processando"):
        try:
            # Extrair embedding (com ou sem TTA)
            embedding = model.extract_embedding_from_path(str(img_path))
            
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
    
    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  Total inserido:  {total_inserted}")
    print(f"  Erros:           {errors}")
    print(f"  Embedding dim:   {embedding_dim}")
    print(f"  TTA:             {'ON' if use_tta else 'OFF'}")
    
    # Stats da collection
    stats = milvus.get_collection_stats()
    print(f"  Collection size: {stats['row_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()