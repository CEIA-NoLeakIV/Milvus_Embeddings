"""
Face Recognition API - Milvus Client
=====================================
Cliente para operações no banco de dados vetorial Milvus Lite.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import MilvusClient as PyMilvusClient, DataType

from .config import Config


class MilvusClient:
    """Cliente para operações no Milvus Lite."""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        Inicializa o cliente Milvus.
        
        Args:
            db_path: Caminho do banco de dados local
            collection_name: Nome da collection
        """
        self.db_path = db_path or Config.MILVUS_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        
        # Conectar ao Milvus Lite
        self.client = PyMilvusClient(self.db_path)
        print(f"✓ Conectado ao Milvus Lite: {self.db_path}")
        
        # Criar collection se não existir
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Garante que a collection existe com o schema correto."""
        if self.client.has_collection(self.collection_name):
            print(f"✓ Collection '{self.collection_name}' já existe.")
            return
        
        # Criar schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        
        # Adicionar campos
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim
        )
        schema.add_field(
            field_name="person_id",
            datatype=DataType.VARCHAR,
            max_length=256
        )
        schema.add_field(
            field_name="image_path",
            datatype=DataType.VARCHAR,
            max_length=512
        )
        schema.add_field(
            field_name="created_at",
            datatype=DataType.VARCHAR,
            max_length=32
        )
        
        # Criar índice para busca
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE"
        )
        
        # Criar collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"✓ Collection '{self.collection_name}' criada!")
    
    def insert(
        self,
        embeddings: List[np.ndarray],
        person_ids: List[str],
        image_paths: List[str]
    ) -> int:
        """
        Insere embeddings no Milvus.
        
        Args:
            embeddings: Lista de embeddings (numpy arrays)
            person_ids: Lista de IDs das pessoas
            image_paths: Lista de caminhos das imagens
            
        Returns:
            Número de registros inseridos
        """
        if not embeddings:
            return 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Preparar dados
        data = []
        for emb, pid, path in zip(embeddings, person_ids, image_paths):
            # Converter numpy para lista se necessário
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            
            data.append({
                "embedding": emb_list,
                "person_id": str(pid),
                "image_path": str(path),
                "created_at": timestamp
            })
        
        # Inserir no Milvus
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        inserted_count = len(data)
        print(f"✓ {inserted_count} embeddings inseridos.")
        
        return inserted_count
    
    def insert_single(
        self,
        embedding: np.ndarray,
        person_id: str,
        image_path: str
    ) -> int:
        """
        Insere um único embedding.
        
        Args:
            embedding: Embedding (numpy array)
            person_id: ID da pessoa
            image_path: Caminho da imagem
            
        Returns:
            Número de registros inseridos (1)
        """
        return self.insert(
            embeddings=[embedding],
            person_ids=[person_id],
            image_paths=[image_path]
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca os embeddings mais similares.
        
        Args:
            query_embedding: Embedding de consulta
            top_k: Número de resultados
            output_fields: Campos a retornar
            
        Returns:
            Lista de resultados com distância e metadados
        """
        if output_fields is None:
            output_fields = ["person_id", "image_path", "created_at"]
        
        # Converter para lista se necessário
        query = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Buscar
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            limit=top_k,
            output_fields=output_fields
        )
        
        # Formatar resultados
        formatted = []
        if results and len(results) > 0:
            for hit in results[0]:
                formatted.append({
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                    "person_id": hit.get("entity", {}).get("person_id"),
                    "image_path": hit.get("entity", {}).get("image_path"),
                    "created_at": hit.get("entity", {}).get("created_at")
                })
        
        return formatted
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da collection.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.client.has_collection(self.collection_name):
            return {"row_count": 0, "exists": False}
        
        stats = self.client.get_collection_stats(self.collection_name)
        info = self.client.describe_collection(self.collection_name)
        
        return {
            "exists": True,
            "row_count": stats.get("row_count", 0),
            "fields": [f["name"] for f in info.get("fields", [])]
        }
    
    def query_by_person(self, person_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Consulta embeddings por person_id.
        
        Args:
            person_id: ID da pessoa
            limit: Limite de resultados
            
        Returns:
            Lista de registros
        """
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'person_id == "{person_id}"',
            output_fields=["id", "person_id", "image_path", "created_at"],
            limit=limit
        )
        return results
    
    def delete_by_person(self, person_id: str) -> int:
        """
        Remove todos os embeddings de uma pessoa.
        
        Args:
            person_id: ID da pessoa
            
        Returns:
            Número de registros removidos
        """
        # Primeiro, buscar os IDs
        results = self.query_by_person(person_id)
        if not results:
            return 0
        
        ids = [r["id"] for r in results]
        
        # Deletar
        self.client.delete(
            collection_name=self.collection_name,
            ids=ids
        )
        
        print(f"✓ {len(ids)} registros de '{person_id}' removidos.")
        return len(ids)
    
    def drop_collection(self):
        """Remove a collection inteira."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"✓ Collection '{self.collection_name}' removida.")
    
    def list_collections(self) -> List[str]:
        """Lista todas as collections."""
        return self.client.list_collections()