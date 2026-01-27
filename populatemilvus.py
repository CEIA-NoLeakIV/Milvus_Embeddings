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
from preprocessing import extract_embedding_standardized

try:
    from utils.face_detection import NoFaceDetectedError, is_face_detection_available
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    NoFaceDetectedError = Exception

LFW_DIR = ROOT_DIR / "lfw"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def parse_args():
    parser = argparse.ArgumentParser(description="Popular Milvus")
    parser.add_argument("--model", type=str, default="mobilenetv3_large", choices=Config.AVAILABLE_MODELS)
    parser.add_argument("--lfw-dir", type=str, default=str(LFW_DIR))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-face-detection", action="store_true")
    parser.add_argument("--skip-no-face", action="store_true")
    parser.add_argument("--face-conf", type=float, default=0.35)
    parser.add_argument("--select-largest", action="store_true", default=True)
    return parser.parse_args()

def get_image_files(directory):
    image_files = []
    path_obj = Path(directory)
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(path_obj.rglob(f"*{ext}"))
        image_files.extend(path_obj.rglob(f"*{ext.upper()}"))
    return sorted(image_files)

def extract_person_id(image_path, base_dir):
    try:
        relative = image_path.relative_to(base_dir)
        return relative.parts[0] if relative.parts else image_path.stem
    except ValueError:
        return image_path.parent.name

def main():
    args = parse_args()
    
    use_tta = not args.no_tta
    embedding_dim = 1024 if use_tta else 512
    use_face_detection = not args.no_face_detection and FACE_DETECTION_AVAILABLE
    
    # Determinar a collection baseada no modelo
    collection_name = Config.get_collection_name(args.model)
    
    print("=" * 60)
    print("  POPULAR MILVUS")
    print("=" * 60)
    print(f"  Modelo:           {args.model}")
    print(f"  Collection:       {collection_name}")
    print(f"  TTA:              {use_tta}")
    print(f"  Face Detection:   {use_face_detection}")
    print("=" * 60)
    
    Config.USE_TTA = use_tta
    Config.EMBEDDING_DIM = embedding_dim
    Config.USE_FACE_DETECTION = use_face_detection
    
    print("\n[1/4] Carregando modelo...")
    model = ModelFactory.create(args.model, use_tta=use_tta)
    
    print("\n[2/4] Conectando ao Milvus...")
    # Conecta especificamente na collection do modelo escolhido
    milvus = MilvusClient(collection_name=collection_name)
    
    if args.recreate:
        print(f"Recriando collection '{collection_name}'...")
        milvus.recreate_collection()
    
    print("\n[3/4] Listando imagens...")
    image_files = get_image_files(args.lfw_dir)
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"\n[4/4] Processando {len(image_files)} imagens...")
    base_dir = Path(args.lfw_dir)
    batch_embeddings = []
    batch_person_ids = []
    batch_image_paths = []
    total_inserted = 0
    
    for img_path in tqdm(image_files, desc="Processando"):
        try:
            embedding = extract_embedding_standardized(
                model,
                file_path=str(img_path),
                use_tta=use_tta,
                use_face_detection=use_face_detection,
                conf_threshold=args.face_conf,
                select_largest=args.select_largest
            )
            person_id = extract_person_id(img_path, base_dir)
            
            batch_embeddings.append(embedding)
            batch_person_ids.append(person_id)
            batch_image_paths.append(str(img_path))
            
            if len(batch_embeddings) >= args.batch_size:
                milvus.insert(batch_embeddings, batch_person_ids, batch_image_paths)
                total_inserted += len(batch_embeddings)
                batch_embeddings, batch_person_ids, batch_image_paths = [], [], []
                
        except (NoFaceDetectedError, Exception) as e:
            if not isinstance(e, NoFaceDetectedError) or not args.skip_no_face:
                pass 
                
    if batch_embeddings:
        milvus.insert(batch_embeddings, batch_person_ids, batch_image_paths)
        total_inserted += len(batch_embeddings)

    print(f"\nSucesso! {total_inserted} imagens inseridas na collection '{collection_name}'.")

if __name__ == "__main__":
    main()