import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .config import Config
from .milvus_client import MilvusClient
from models import ModelFactory

# IMPORTANTE: Importar o módulo de preprocessing centralizado
from preprocessing import extract_embedding_standardized

# Importar exceção de detecção facial
try:
    from utils.face_detection import NoFaceDetectedError, is_face_detection_available
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    class NoFaceDetectedError(Exception):
        pass


# ===========================================
# Variáveis globais (lazy loading)
# ===========================================
_models = {}
_milvus_clients = {}  # Agora é um dicionário para suportar múltiplas collections


def get_model(model_name: str):
    """Retorna o modelo (lazy loading)."""
    global _models
    
    if model_name not in _models:
        print(f"Carregando modelo: {model_name}...")
        _models[model_name] = ModelFactory.create(
            model_name,
            use_tta=Config.USE_TTA
        )
        print(f"✓ Modelo '{model_name}' carregado! (TTA={Config.USE_TTA})")
    
    return _models[model_name]


def get_milvus_client(collection_name: str = None):
    """Retorna o cliente Milvus para uma collection específica."""
    global _milvus_clients
    
    # Se não passar nome, usa o padrão do Config (fallback)
    if collection_name is None:
        collection_name = Config.COLLECTION_NAME
        
    if collection_name not in _milvus_clients:
        print(f"Conectando ao Milvus (Collection: {collection_name})...")
        _milvus_clients[collection_name] = MilvusClient(collection_name=collection_name)
    
    return _milvus_clients[collection_name]


# ===========================================
# Factory da aplicação Flask
# ===========================================
def create_app():
    """Cria e configura a aplicação Flask."""
    
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    CORS(app)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Verifica o status da API."""
        milvus_ok = True
        # Verifica conexão básica com qualquer client já instanciado ou cria um padrão
        try:
            client = get_milvus_client(Config.COLLECTION_NAME)
            stats = client.get_collection_stats()
        except Exception:
            milvus_ok = False
        
        face_detection_status = {
            "available": FACE_DETECTION_AVAILABLE,
            "enabled": Config.USE_FACE_DETECTION,
            "conf_threshold": Config.FACE_DETECTION_CONF_THRESHOLD
        }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": list(_models.keys()),
            "models_available": Config.AVAILABLE_MODELS,
            "milvus_connected": milvus_ok,
            "device": str(Config.DEVICE),
            "use_tta": Config.USE_TTA,
            "face_detection": face_detection_status
        })
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """Lista os modelos e suas collections."""
        models_info = []
        
        for model_name in Config.AVAILABLE_MODELS:
            weight_path = Config.get_model_weight_path(model_name)
            weight_exists = weight_path.exists()
            collection_name = Config.get_collection_name(model_name)
            
            models_info.append({
                "name": model_name,
                "collection": collection_name,
                "embedding_dim": Config.EMBEDDING_DIM,
                "weight_exists": weight_exists,
                "weight_path": str(weight_path),
                "loaded": model_name in _models
            })
        
        return jsonify({
            "models": models_info,
            "default": Config.DEFAULT_MODEL,
            "use_tta": Config.USE_TTA,
            "face_detection_enabled": Config.USE_FACE_DETECTION
        })
    
    # ===========================================
    # Rotas de Embedding
    # ===========================================
    @app.route('/api/embedding', methods=['POST'])
    def generate_embedding():
        if 'image' not in request.files:
            return jsonify({"error": "Campo 'image' é obrigatório"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400
        
        if not Config.is_allowed_file(file.filename):
            return jsonify({"error": "Extensão não permitida"}), 400
        
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400
        
        use_face_detection_param = request.form.get('use_face_detection')
        use_face_detection = use_face_detection_param.lower() == 'true' if use_face_detection_param is not None else Config.USE_FACE_DETECTION
        
        try:
            model = get_model(model_name)
            file_bytes = file.read()
            
            embedding = extract_embedding_standardized(
                model,
                file_bytes=file_bytes,
                use_tta=Config.USE_TTA,
                use_face_detection=use_face_detection,
                conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                select_largest=Config.FACE_DETECTION_SELECT_LARGEST
            )
            
            return jsonify({
                "success": True,
                "model": model_name,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding),
                "filename": secure_filename(file.filename),
                "face_detection_used": use_face_detection
            })
        except NoFaceDetectedError as e:
            return jsonify({
                "success": False,
                "error": "no_face_detected",
                "message": "Nenhuma face foi detectada na imagem",
                "details": str(e)
            }), 422
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/embeddings/batch', methods=['POST'])
    def generate_embeddings_batch():
        if 'images' not in request.files:
            return jsonify({"error": "Campo 'images' é obrigatório"}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
            
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400
            
        use_face_detection_param = request.form.get('use_face_detection')
        use_face_detection = use_face_detection_param.lower() == 'true' if use_face_detection_param is not None else Config.USE_FACE_DETECTION
        
        try:
            model = get_model(model_name)
            results = []
            successful = 0
            failed = 0
            no_face_count = 0
            
            for file in files:
                filename = secure_filename(file.filename)
                try:
                    if not Config.is_allowed_file(filename):
                        results.append({"filename": filename, "success": False, "error": "Extensão não permitida"})
                        failed += 1
                        continue
                        
                    file_bytes = file.read()
                    embedding = extract_embedding_standardized(
                        model,
                        file_bytes=file_bytes,
                        use_tta=Config.USE_TTA,
                        use_face_detection=use_face_detection,
                        conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                        select_largest=Config.FACE_DETECTION_SELECT_LARGEST
                    )
                    results.append({"filename": filename, "success": True, "embedding": embedding.tolist()})
                    successful += 1
                except NoFaceDetectedError:
                    results.append({"filename": filename, "success": False, "error": "no_face_detected"})
                    failed += 1
                    no_face_count += 1
                except Exception as e:
                    results.append({"filename": filename, "success": False, "error": str(e)})
                    failed += 1
            
            return jsonify({
                "success": True,
                "model": model_name,
                "results": results,
                "total": len(files),
                "successful": successful,
                "failed": failed,
                "no_face_detected": no_face_count
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ===========================================
    # Rotas do Milvus (Insert/Search)
    # ===========================================
    @app.route('/api/milvus/insert', methods=['POST'])
    def insert_to_milvus():
        if 'images' not in request.files:
            return jsonify({"error": "Campo 'images' é obrigatório"}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        person_id = request.form.get('person_id')
        if not person_id:
            return jsonify({"error": "Campo 'person_id' é obrigatório"}), 400
        
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400
        
        use_face_detection_param = request.form.get('use_face_detection')
        use_face_detection = use_face_detection_param.lower() == 'true' if use_face_detection_param is not None else Config.USE_FACE_DETECTION

        image_paths_json = request.form.get('image_paths')
        original_paths = json.loads(image_paths_json) if image_paths_json else None
        
        try:
            model = get_model(model_name)
            
            # Obter nome da collection baseado no modelo
            collection_name = Config.get_collection_name(model_name)
            client = get_milvus_client(collection_name)
            
            embeddings = []
            person_ids = []
            paths = []
            errors = []
            no_face_errors = []
            
            for idx, file in enumerate(files):
                filename = secure_filename(file.filename)
                try:
                    if not Config.is_allowed_file(filename):
                        errors.append(f"{filename}: extensão não permitida")
                        continue
                    
                    file_bytes = file.read()
                    embedding = extract_embedding_standardized(
                        model,
                        file_bytes=file_bytes,
                        use_tta=Config.USE_TTA,
                        use_face_detection=use_face_detection,
                        conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                        select_largest=Config.FACE_DETECTION_SELECT_LARGEST
                    )
                    
                    embeddings.append(embedding)
                    person_ids.append(person_id)
                    
                    if original_paths and idx < len(original_paths):
                        paths.append(original_paths[idx])
                    else:
                        paths.append(filename)
                
                except NoFaceDetectedError:
                    no_face_errors.append(filename)
                    errors.append(f"{filename}: nenhuma face detectada")
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
            
            if embeddings:
                inserted = client.insert(embeddings=embeddings, person_ids=person_ids, image_paths=paths)
            else:
                inserted = 0
            
            response = {
                "success": True,
                "message": f"{inserted} embeddings inseridos com sucesso",
                "inserted_count": inserted,
                "collection": collection_name,
                "model": model_name,
                "person_id": person_id
            }
            if errors:
                response["errors"] = errors
                response["no_face_count"] = len(no_face_errors)
            
            return jsonify(response)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/milvus/search', methods=['POST'])
    def search_milvus():
        if 'image' not in request.files:
            return jsonify({"error": "Campo 'image' é obrigatório"}), 400
        
        file = request.files['image']
        if not Config.is_allowed_file(file.filename):
            return jsonify({"error": "Extensão não permitida"}), 400
        
        model_name = request.form.get('model', Config.DEFAULT_MODEL)
        top_k = int(request.form.get('top_k', 5))
        
        if model_name not in Config.AVAILABLE_MODELS:
            return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400
        
        use_face_detection_param = request.form.get('use_face_detection')
        use_face_detection = use_face_detection_param.lower() == 'true' if use_face_detection_param is not None else Config.USE_FACE_DETECTION
        
        try:
            model = get_model(model_name)
            
            # Obter nome da collection baseado no modelo
            collection_name = Config.get_collection_name(model_name)
            client = get_milvus_client(collection_name)
            
            file_bytes = file.read()
            embedding = extract_embedding_standardized(
                model,
                file_bytes=file_bytes,
                use_tta=Config.USE_TTA,
                use_face_detection=use_face_detection,
                conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                select_largest=Config.FACE_DETECTION_SELECT_LARGEST
            )
            
            results = client.search(embedding, top_k=top_k)
            
            return jsonify({
                "success": True,
                "model": model_name,
                "collection": collection_name,
                "query_filename": secure_filename(file.filename),
                "top_k": top_k,
                "results": results
            })
        
        except NoFaceDetectedError as e:
            return jsonify({
                "success": False,
                "error": "no_face_detected",
                "message": "Nenhuma face foi detectada na imagem de busca",
                "details": str(e)
            }), 422
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/milvus/stats', methods=['GET'])
    def milvus_stats():
        """Retorna estatísticas do Milvus (todas collections)."""
        try:
            all_stats = {}
            # Itera sobre os modelos para pegar todas as collections
            for model_name in Config.AVAILABLE_MODELS:
                coll_name = Config.get_collection_name(model_name)
                try:
                    client = get_milvus_client(coll_name)
                    stats = client.get_collection_stats()
                    all_stats[coll_name] = stats
                except Exception:
                    all_stats[coll_name] = {"error": "not initialized or empty"}

            return jsonify({
                "success": True,
                "collections": all_stats
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # Error handlers (mantidos iguais)
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "Arquivo muito grande", "max_size_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024)}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint não encontrado"}), 404
    
    @app.errorhandler(422)
    def unprocessable_entity(e):
        return jsonify({"error": "Conteúdo não processável", "message": "A imagem não pôde ser processada."}), 422
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Erro interno do servidor"}), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)