"""
Face Recognition API - Milvus Client
=====================================
Cliente para operações no banco de dados vetorial Milvus Lite.
Suporta embeddings de 1024 dimensões (com TTA).
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
        self.embedding_dim = Config.EMBEDDING_DIM  # 1024 com TTA
        
        # Conectar ao Milvus Lite
        self.client = PyMilvusClient(self.db_path)
        print(f"✓ Conectado ao Milvus Lite: {self.db_path}")
        print(f"  Embedding dim: {self.embedding_dim}")
        
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
            dim=self.embedding_dim  # 1024 com TTA
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
        print(f"✓ Collection '{self.collection_name}' criada com dim={self.embedding_dim}!")
    
    def insert(
        self,
        embeddings: List[np.ndarray],
        person_ids: List[str],
        image_paths: List[str]
    ) -> int:
        """
        Insere embeddings no Milvus.
        
        Args:
            embeddings: Lista de embeddings (numpy arrays de 1024 dims)
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
            embedding: Embedding (numpy array de 1024 dims)
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
            query_embedding: Embedding de consulta (1024 dims)
            top_k: Número de resultados
            output_fields: Campos a retornar
            
        Returns:
            Lista de resultados com distância e campos
        """
        if output_fields is None:
            output_fields = ["person_id", "image_path", "created_at"]
        
        # Converter para lista se necessário
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_list],
            limit=top_k,
            output_fields=output_fields
        )
        
        return results[0] if results else []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da collection.
        
        Returns:
            Dict com estatísticas
        """
        stats = self.client.get_collection_stats(self.collection_name)
        info = self.client.describe_collection(self.collection_name)
        
        return {
            "collection_name": self.collection_name,
            "row_count": stats.get("row_count", 0),
            "fields": info.get("fields", []),
            "embedding_dim": self.embedding_dim
        }
    
    def delete_collection(self):
        """Remove a collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            print(f"✓ Collection '{self.collection_name}' removida.")
    
    def recreate_collection(self):
        """Recria a collection (apaga dados existentes)."""
        self.delete_collection()
        self._ensure_collection()
    
    def query(
        self,
        filter_expr: str = "",
        output_fields: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Consulta registros com filtro.
        
        Args:
            filter_expr: Expressão de filtro
            output_fields: Campos a retornar
            limit: Número máximo de resultados
            
        Returns:
            Lista de registros
        """
        if output_fields is None:
            output_fields = ["id", "person_id", "image_path", "created_at"]
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit
        )
        
        return results