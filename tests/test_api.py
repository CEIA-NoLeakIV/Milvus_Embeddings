"""
Face Recognition API - Tests
=============================
Testes para validar os endpoints da API.

Uso:
    pytest tests/test_api.py -v
    python -m pytest tests/test_api.py -v
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from io import BytesIO

import pytest
import numpy as np
from PIL import Image

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.api import create_app
from app.config import Config


# ===========================================
# Fixtures
# ===========================================
@pytest.fixture
def app():
    """Cria instância da aplicação para testes."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Cria cliente de teste."""
    return app.test_client()


@pytest.fixture
def sample_image():
    """Cria uma imagem de teste em memória."""
    # Criar imagem RGB 112x112 (tamanho esperado pelos modelos)
    img_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Salvar em buffer
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    
    return buffer


@pytest.fixture
def sample_image_file(sample_image):
    """Retorna tupla (BytesIO, filename) para upload."""
    return (sample_image, 'test_image.jpg')


def create_test_image(width=112, height=112):
    """Cria uma imagem de teste."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    
    return buffer


# ===========================================
# Testes de Health Check
# ===========================================
class TestHealthCheck:
    """Testes para o endpoint /api/health."""
    
    def test_health_check_returns_200(self, client):
        """Verifica se o health check retorna 200."""
        response = client.get('/api/health')
        assert response.status_code == 200
    
    def test_health_check_returns_json(self, client):
        """Verifica se retorna JSON válido."""
        response = client.get('/api/health')
        data = response.get_json()
        
        assert data is not None
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_health_check_contains_required_fields(self, client):
        """Verifica se contém todos os campos obrigatórios."""
        response = client.get('/api/health')
        data = response.get_json()
        
        required_fields = ['status', 'timestamp', 'models_available', 'device']
        for field in required_fields:
            assert field in data, f"Campo '{field}' não encontrado"


# ===========================================
# Testes de Listagem de Modelos
# ===========================================
class TestModels:
    """Testes para o endpoint /api/models."""
    
    def test_list_models_returns_200(self, client):
        """Verifica se a listagem retorna 200."""
        response = client.get('/api/models')
        assert response.status_code == 200
    
    def test_list_models_contains_available_models(self, client):
        """Verifica se lista os modelos disponíveis."""
        response = client.get('/api/models')
        data = response.get_json()
        
        assert 'models' in data
        assert isinstance(data['models'], list)
        assert len(data['models']) > 0
    
    def test_list_models_has_default(self, client):
        """Verifica se há um modelo padrão definido."""
        response = client.get('/api/models')
        data = response.get_json()
        
        assert 'default' in data
        assert data['default'] in Config.AVAILABLE_MODELS


# ===========================================
# Testes de Geração de Embedding
# ===========================================
class TestEmbedding:
    """Testes para o endpoint /api/embedding."""
    
    def test_embedding_without_image_returns_400(self, client):
        """Verifica erro quando não envia imagem."""
        response = client.post('/api/embedding')
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
    
    def test_embedding_with_invalid_extension_returns_400(self, client):
        """Verifica erro com extensão inválida."""
        # Criar arquivo com extensão inválida
        buffer = BytesIO(b'fake content')
        
        response = client.post(
            '/api/embedding',
            data={'image': (buffer, 'test.txt')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_embedding_with_invalid_model_returns_400(self, client):
        """Verifica erro com modelo inexistente."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/embedding',
            data={
                'image': (buffer, 'test.jpg'),
                'model': 'modelo_inexistente'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    @pytest.mark.skipif(
        not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
        reason="Peso do modelo não encontrado"
    )
    def test_embedding_with_valid_image_returns_200(self, client):
        """Verifica geração de embedding com imagem válida."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/embedding',
            data={
                'image': (buffer, 'test.jpg'),
                'model': 'mobilenetv3_large'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'embedding' in data
        assert len(data['embedding']) == Config.EMBEDDING_DIM
    
    @pytest.mark.skipif(
        not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
        reason="Peso do modelo não encontrado"
    )
    def test_embedding_returns_correct_dimension(self, client):
        """Verifica se o embedding tem a dimensão correta (512)."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/embedding',
            data={
                'image': (buffer, 'test.jpg'),
                'model': 'mobilenetv3_large'
            },
            content_type='multipart/form-data'
        )
        
        data = response.get_json()
        
        if data.get('success'):
            assert data['embedding_dim'] == 512
            assert len(data['embedding']) == 512


# ===========================================
# Testes de Embedding em Lote
# ===========================================
class TestBatchEmbedding:
    """Testes para o endpoint /api/embeddings/batch."""
    
    def test_batch_without_images_returns_400(self, client):
        """Verifica erro quando não envia imagens."""
        response = client.post('/api/embeddings/batch')
        assert response.status_code == 400
    
    def test_batch_with_invalid_model_returns_400(self, client):
        """Verifica erro com modelo inexistente."""
        buffer1 = create_test_image()
        buffer2 = create_test_image()
        
        response = client.post(
            '/api/embeddings/batch',
            data={
                'images': [
                    (buffer1, 'test1.jpg'),
                    (buffer2, 'test2.jpg')
                ],
                'model': 'modelo_inexistente'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
    
    @pytest.mark.skipif(
        not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
        reason="Peso do modelo não encontrado"
    )
    def test_batch_with_valid_images_returns_200(self, client):
        """Verifica geração de embeddings em lote."""
        buffer1 = create_test_image()
        buffer2 = create_test_image()
        
        response = client.post(
            '/api/embeddings/batch',
            data={
                'images': [
                    (buffer1, 'test1.jpg'),
                    (buffer2, 'test2.jpg')
                ],
                'model': 'mobilenetv3_large'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'results' in data
        assert data['total'] == 2


# ===========================================
# Testes de Inserção no Milvus
# ===========================================
class TestMilvusInsert:
    """Testes para o endpoint /api/milvus/insert."""
    
    def test_insert_without_images_returns_400(self, client):
        """Verifica erro quando não envia imagens."""
        response = client.post('/api/milvus/insert')
        assert response.status_code == 400
    
    def test_insert_without_person_id_returns_400(self, client):
        """Verifica erro quando não envia person_id."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/milvus/insert',
            data={
                'images': (buffer, 'test.jpg')
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
        assert 'person_id' in data['error'].lower()
    
    @pytest.mark.skipif(
        not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
        reason="Peso do modelo não encontrado"
    )
    def test_insert_with_valid_data_returns_200(self, client):
        """Verifica inserção com dados válidos."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/milvus/insert',
            data={
                'images': (buffer, 'test.jpg'),
                'person_id': 'test_person_123',
                'model': 'mobilenetv3_large'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert data['inserted_count'] >= 1


# ===========================================
# Testes de Busca no Milvus
# ===========================================
class TestMilvusSearch:
    """Testes para o endpoint /api/milvus/search."""
    
    def test_search_without_image_returns_400(self, client):
        """Verifica erro quando não envia imagem."""
        response = client.post('/api/milvus/search')
        assert response.status_code == 400
    
    @pytest.mark.skipif(
        not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
        reason="Peso do modelo não encontrado"
    )
    def test_search_with_valid_image_returns_200(self, client):
        """Verifica busca com imagem válida."""
        buffer = create_test_image()
        
        response = client.post(
            '/api/milvus/search',
            data={
                'image': (buffer, 'test.jpg'),
                'model': 'mobilenetv3_large',
                'top_k': '5'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'results' in data


# ===========================================
# Testes de Estatísticas do Milvus
# ===========================================
class TestMilvusStats:
    """Testes para o endpoint /api/milvus/stats."""
    
    def test_stats_returns_200(self, client):
        """Verifica se retorna estatísticas."""
        response = client.get('/api/milvus/stats')
        assert response.status_code == 200
    
    def test_stats_contains_collection_info(self, client):
        """Verifica se contém informações da collection."""
        response = client.get('/api/milvus/stats')
        data = response.get_json()
        
        assert 'success' in data
        assert 'collection' in data


# ===========================================
# Testes de Error Handlers
# ===========================================
class TestErrorHandlers:
    """Testes para os handlers de erro."""
    
    def test_404_returns_json(self, client):
        """Verifica se 404 retorna JSON."""
        response = client.get('/api/endpoint_inexistente')
        assert response.status_code == 404
        
        data = response.get_json()
        assert 'error' in data


# ===========================================
# Testes de Integração
# ===========================================
@pytest.mark.skipif(
    not Config.MODEL_WEIGHTS.get('mobilenetv3_large', Path()).exists(),
    reason="Peso do modelo não encontrado"
)
class TestIntegration:
    """Testes de integração completa."""
    
    def test_full_flow_insert_and_search(self, client):
        """Testa fluxo completo: inserir e buscar."""
        # 1. Inserir uma imagem
        buffer1 = create_test_image()
        
        insert_response = client.post(
            '/api/milvus/insert',
            data={
                'images': (buffer1, 'person1.jpg'),
                'person_id': 'integration_test_person',
                'model': 'mobilenetv3_large'
            },
            content_type='multipart/form-data'
        )
        
        assert insert_response.status_code == 200
        insert_data = insert_response.get_json()
        assert insert_data['success'] is True
        
        # 2. Buscar com outra imagem
        buffer2 = create_test_image()
        
        search_response = client.post(
            '/api/milvus/search',
            data={
                'image': (buffer2, 'query.jpg'),
                'model': 'mobilenetv3_large',
                'top_k': '5'
            },
            content_type='multipart/form-data'
        )
        
        assert search_response.status_code == 200
        search_data = search_response.get_json()
        assert search_data['success'] is True
        assert 'results' in search_data


# ===========================================
# Main
# ===========================================
if __name__ == '__main__':
    pytest.main([__file__, '-v'])