#!/usr/bin/env python3
"""
Script para popular o Milvus com embeddings do dataset LFW
==========================================================

Este script:
1. Carrega o modelo MobileNetV3 Large
2. Percorre todas as imagens do dataset LFW
3. Extrai os embeddings de cada imagem
4. Insere no Milvus Lite (salvo em data/milvus_face.db)

Uso:
    python populate_milvus_lfw.py
    python populate_milvus_lfw.py --model cosface_resnet50
    python populate_milvus_lfw.py --batch-size 64 --limit 1000
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
    No LFW, a estrutura é: lfw/Nome_Pessoa/Nome_Pessoa_0001.jpg
    """
    try:
        # Pega o nome da pasta pai (nome da pessoa)
        return image_path.parent.name
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    
    print("=" * 60)
    print("       POPULAR MILVUS COM LFW")
    print("=" * 60)
    print(f"  Modelo:      {args.model}")
    print(f"  LFW Dir:     {args.lfw_dir}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  Limite:      {args.limit or 'Todas'}")
    print(f"  Device:      {Config.DEVICE}")
    print("=" * 60)
    print()
    
    # Verificar diretório LFW
    lfw_path = Path(args.lfw_dir)
    if not lfw_path.exists():
        print(f"❌ Erro: Diretório LFW não encontrado: {lfw_path}")
        print(f"   Certifique-se de que a pasta 'lfw' está na raiz do projeto.")
        sys.exit(1)
    
    # Listar imagens
    print("📂 Buscando imagens...")
    image_files = get_image_files(args.lfw_dir)
    
    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em: {args.lfw_dir}")
        sys.exit(1)
    
    print(f"✓ Encontradas {len(image_files)} imagens")
    
    # Aplicar limite se especificado
    if args.limit:
        image_files = image_files[:args.limit]
        print(f"✓ Limitado a {len(image_files)} imagens")
    
    # Carregar modelo
    print(f"\n🔄 Carregando modelo {args.model}...")
    try:
        model = ModelFactory.create(args.model)
        print(f"✓ Modelo carregado com sucesso!")
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print(f"   Verifique se o arquivo de pesos existe em models/weights/")
        sys.exit(1)
    
    # Conectar ao Milvus
    print(f"\n🔄 Conectando ao Milvus...")
    client = MilvusClient()
    
    # Recriar collection se solicitado
    if args.recreate:
        print("⚠️  Recriando collection...")
        client.drop_collection()
        client = MilvusClient()  # Reconectar para criar nova collection
    
    # Verificar estatísticas atuais
    stats = client.get_collection_stats()
    print(f"✓ Collection: {client.collection_name}")
    print(f"✓ Registros atuais: {stats.get('row_count', 0)}")
    
    # Processar imagens em lotes
    print(f"\n🚀 Iniciando extração e inserção...")
    print("-" * 60)
    
    batch_embeddings = []
    batch_person_ids = []
    batch_image_paths = []
    
    total_inserted = 0
    total_errors = 0
    
    start_time = datetime.now()
    
    for img_path in tqdm(image_files, desc="Processando"):
        try:
            # Abrir imagem
            image = Image.open(img_path).convert('RGB')
            
            # Extrair embedding
            embedding = model.extract_embedding_from_pil(image)
            
            # Extrair person_id (nome da pasta)
            person_id = extract_person_id(img_path, lfw_path)
            
            # Adicionar ao lote
            batch_embeddings.append(embedding)
            batch_person_ids.append(person_id)
            batch_image_paths.append(str(img_path))
            
            # Inserir quando atingir o tamanho do lote
            if len(batch_embeddings) >= args.batch_size:
                inserted = client.insert(
                    embeddings=batch_embeddings,
                    person_ids=batch_person_ids,
                    image_paths=batch_image_paths
                )
                total_inserted += inserted
                
                # Limpar lote
                batch_embeddings = []
                batch_person_ids = []
                batch_image_paths = []
                
        except Exception as e:
            total_errors += 1
            # Descomente para debug:
            # print(f"\n⚠️  Erro em {img_path.name}: {e}")
    
    # Inserir o que sobrou no último lote
    if batch_embeddings:
        inserted = client.insert(
            embeddings=batch_embeddings,
            person_ids=batch_person_ids,
            image_paths=batch_image_paths
        )
        total_inserted += inserted
    
    # Estatísticas finais
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 60)
    print("       CONCLUÍDO!")
    print("=" * 60)
    print(f"  ✓ Total inserido:  {total_inserted} embeddings")
    print(f"  ✗ Erros:           {total_errors}")
    print(f"  ⏱️  Tempo total:    {duration:.2f} segundos")
    print(f"  📊 Velocidade:     {total_inserted / duration:.2f} imgs/seg")
    print()
    
    # Verificar estatísticas finais
    final_stats = client.get_collection_stats()
    print(f"  📁 Total na collection: {final_stats.get('row_count', 0)}")
    print(f"  💾 Banco salvo em: {Config.MILVUS_DB_PATH}")
    print("=" * 60)
    
    # Exemplo de query
    print("\n📋 Amostra dos dados inseridos:")
    print("-" * 60)
    
    try:
        sample = client.client.query(
            collection_name=client.collection_name,
            filter="",
            output_fields=["id", "person_id", "image_path", "created_at"],
            limit=5
        )
        
        for i, record in enumerate(sample, 1):
            print(f"  {i}. ID: {record['id']}")
            print(f"     Person: {record['person_id']}")
            print(f"     Path: {Path(record['image_path']).name}")
            print()
    except Exception as e:
        print(f"  Erro ao buscar amostra: {e}")


if __name__ == "__main__":
    main()