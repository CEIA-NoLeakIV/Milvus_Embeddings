"""
Face Recognition API - Flask Application
=========================================
API REST para extração de embeddings e inserção no Milvus.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .config import Config
from .milvus_client import MilvusClient
from models import ModelFactory


# ===========================================
# Variáveis globais (lazy loading)
# ===========================================
_models = {}
_milvus_client = None


def get_model(model_name: str):
    """Retorna o modelo (lazy loading)."""
    global _models
    
    if model_name not in _models:
        print(f"Carregando modelo: {model_name}...")
        _models[model_name] = ModelFactory.create(model_name)
        print(f"✓ Modelo '{model_name}' carregado!")
    
    return _models[model_name]


def get_milvus_client():
    """Retorna o cliente Milvus (singleton)."""
    global _milvus_client
    
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    
    return _milvus_client


# ===========================================
# Factory da aplicação Flask
# ===========================================
def create_app():
    """Cria e configura a aplicação Flask."""
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Habilitar CORS
    CORS(app)
    
    # ===========================================
    # Rotas de Health Check
    # ===========================================
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Verifica o status da API."""
        try:
            client = get_milvus_client()
            stats = client.get_collection_stats()
            milvus_ok = stats.get("exists", False) or stats.get("row_count", 0) >= 0
        except Exception as e:
            milvus_ok = False
        
        # Verificar modelos carregados
        loaded_models = list(_models.keys())
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": loaded_models,
            "models_available": Config.AVAILABLE_MODELS,
            "milvus_connected": milvus_ok,
            "device": str(Config.DEVICE)
        })
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """Lista os modelos disponíveis."""
        models_info = []
        
        for model_name in Config.AVAILABLE_MODELS:
            weight_path = Config.get_model_weight_path(model_name)
            weight_exists = weight_path.exists()
            
            models_info.append({
                "name": model_name,
                "embedding_dim": Config.EMBEDDING_DIM,
                "weight_exists": weight_exists,
                "weight_path": str(weight_path),
                "loaded": model_name in _models
            })
        
        return jsonify({
            "models": models_info,
            "default": Config.DEFAULT_MODEL
        })
    
    # ===========================================
    # Rotas de Embedding
    # ===========================================
    @app.route('/api/embedding', methods=['POST'])
    def generate_embedding():
        """
        Gera embedding de uma única imagem.
        
        Form Data:
            - image: arquivo de imagem
            - model: nome do modelo (opcional, default: mobilenetv3_large)
        """
        # Validar arquivo
        if 'image' not in request.files:
            return jsonify({"error": "Campo 'image' é obrigatório"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400
        
        if not Config.is_allowed_file(file.filename):
            return jsonify({
                "error": f"Extensão não permitida. Use: {Config.ALLOWED_EXTENSIONS}"
            }), 400
        
        # Obter modelo
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({
                "error": f"Modelo '{model_name}' não encontrado. "
                        f"Disponíveis: {Config.AVAILABLE_MODELS}"
            }), 400
        
        try:
            # Carregar modelo
            model = get_model(model_name)
            
            # Abrir imagem
            image = Image.open(file.stream).convert('RGB')
            
            # Extrair embedding
            embedding = model.extract_embedding_from_pil(image)
            
            return jsonify({
                "success": True,
                "model": model_name,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
                "filename": secure_filename(file.filename)
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/embeddings/batch', methods=['POST'])
    def generate_embeddings_batch():
        """
        Gera embeddings para múltiplas imagens.
        
        Form Data:
            - images: múltiplos arquivos de imagem
            - model: nome do modelo (opcional)
        """
        # Validar arquivos
        if 'images' not in request.files:
            return jsonify({"error": "Campo 'images' é obrigatório"}), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        if len(files) > Config.MAX_BATCH_SIZE:
            return jsonify({
                "error": f"Máximo de {Config.MAX_BATCH_SIZE} imagens por lote"
            }), 400
        
        # Obter modelo
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({
                "error": f"Modelo '{model_name}' não encontrado"
            }), 400
        
        try:
            model = get_model(model_name)
            
            results = []
            successful = 0
            failed = 0
            
            for file in files:
                filename = secure_filename(file.filename)
                
                try:
                    if not Config.is_allowed_file(filename):
                        results.append({
                            "filename": filename,
                            "success": False,
                            "error": "Extensão não permitida"
                        })
                        failed += 1
                        continue
                    
                    # Processar imagem
                    image = Image.open(file.stream).convert('RGB')
                    embedding = model.extract_embedding_from_pil(image)
                    
                    results.append({
                        "filename": filename,
                        "success": True,
                        "embedding": embedding.tolist()
                    })
                    successful += 1
                    
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": str(e)
                    })
                    failed += 1
            
            return jsonify({
                "success": True,
                "model": model_name,
                "results": results,
                "total": len(files),
                "successful": successful,
                "failed": failed
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # ===========================================
    # Rotas do Milvus
    # ===========================================
    @app.route('/api/milvus/insert', methods=['POST'])
    def insert_to_milvus():
        """
        Gera embeddings e insere no Milvus.
        
        Form Data:
            - images: arquivos de imagem
            - model: nome do modelo (opcional)
            - person_id: ID da pessoa (obrigatório)
            - image_paths: JSON array com caminhos originais (opcional)
        """
        # Validar arquivos
        if 'images' not in request.files:
            return jsonify({"error": "Campo 'images' é obrigatório"}), 400
        
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        # Validar person_id
        person_id = request.form.get('person_id')
        if not person_id:
            return jsonify({"error": "Campo 'person_id' é obrigatório"}), 400
        
        # Obter modelo
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({
                "error": f"Modelo '{model_name}' não encontrado"
            }), 400
        
        # Obter caminhos originais (opcional)
        image_paths_json = request.form.get('image_paths')
        if image_paths_json:
            try:
                original_paths = json.loads(image_paths_json)
            except json.JSONDecodeError:
                original_paths = None
        else:
            original_paths = None
        
        try:
            model = get_model(model_name)
            client = get_milvus_client()
            
            embeddings = []
            person_ids = []
            paths = []
            errors = []
            
            for idx, file in enumerate(files):
                filename = secure_filename(file.filename)
                
                try:
                    if not Config.is_allowed_file(filename):
                        errors.append(f"{filename}: extensão não permitida")
                        continue
                    
                    # Processar imagem
                    image = Image.open(file.stream).convert('RGB')
                    embedding = model.extract_embedding_from_pil(image)
                    
                    embeddings.append(embedding)
                    person_ids.append(person_id)
                    
                    # Usar caminho original ou nome do arquivo
                    if original_paths and idx < len(original_paths):
                        paths.append(original_paths[idx])
                    else:
                        paths.append(filename)
                    
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
            
            # Inserir no Milvus
            if embeddings:
                inserted = client.insert(
                    embeddings=embeddings,
                    person_ids=person_ids,
                    image_paths=paths
                )
            else:
                inserted = 0
            
            response = {
                "success": True,
                "message": f"{inserted} embeddings inseridos com sucesso",
                "inserted_count": inserted,
                "collection": client.collection_name,
                "model": model_name,
                "person_id": person_id
            }
            
            if errors:
                response["errors"] = errors
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/milvus/search', methods=['POST'])
    def search_milvus():
        """
        Busca faces similares no Milvus.
        
        Form Data:
            - image: arquivo de imagem para busca
            - model: nome do modelo (opcional)
            - top_k: número de resultados (opcional, default: 5)
        """
        # Validar arquivo
        if 'image' not in request.files:
            return jsonify({"error": "Campo 'image' é obrigatório"}), 400
        
        file = request.files['image']
        
        if not Config.is_allowed_file(file.filename):
            return jsonify({"error": "Extensão não permitida"}), 400
        
        # Obter parâmetros
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        top_k = int(request.form.get('top_k', 5))
        
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({
                "error": f"Modelo '{model_name}' não encontrado"
            }), 400
        
        try:
            model = get_model(model_name)
            client = get_milvus_client()
            
            # Extrair embedding
            image = Image.open(file.stream).convert('RGB')
            embedding = model.extract_embedding_from_pil(image)
            
            # Buscar similares
            results = client.search(embedding, top_k=top_k)
            
            return jsonify({
                "success": True,
                "model": model_name,
                "query_filename": secure_filename(file.filename),
                "top_k": top_k,
                "results": results
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/milvus/stats', methods=['GET'])
    def milvus_stats():
        """Retorna estatísticas do Milvus."""
        try:
            client = get_milvus_client()
            stats = client.get_collection_stats()
            collections = client.list_collections()
            
            return jsonify({
                "success": True,
                "collection": client.collection_name,
                "stats": stats,
                "all_collections": collections
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/milvus/query/<person_id>', methods=['GET'])
    def query_person(person_id: str):
        """Consulta embeddings por person_id."""
        try:
            client = get_milvus_client()
            results = client.query_by_person(person_id)
            
            return jsonify({
                "success": True,
                "person_id": person_id,
                "count": len(results),
                "results": results
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # ===========================================
    # Error Handlers
    # ===========================================
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({
            "error": "Arquivo muito grande",
            "max_size_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        }), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint não encontrado"}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Erro interno do servidor"}), 500
    
    return app


# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)