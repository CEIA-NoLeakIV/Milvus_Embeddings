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
        """
        self.db_path = db_path or Config.MILVUS_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        
        # Conectar ao Milvus Lite
        try:
            self.client = PyMilvusClient(self.db_path)
            print(f"✓ Conectado ao Milvus Lite: {self.db_path}")
            print(f"  Embedding dim: {self.embedding_dim}")
        except Exception as e:
            print(f"✗ Erro ao conectar no Milvus: {e}")
            raise e
        
        # Criar collection se não existir
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Garante que a collection existe com o schema correto."""
        if self.client.has_collection(self.collection_name):
            # print(f"✓ Collection '{self.collection_name}' já existe.")
            return
        
        # Criar schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        
        # Adicionar campos
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(field_name="person_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=32)
        
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
        """Insere embeddings no Milvus."""
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
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        inserted_count = len(data)
        # print(f"✓ {inserted_count} embeddings inseridos.")
        return inserted_count
    
    def insert_single(self, embedding: np.ndarray, person_id: str, image_path: str) -> int:
        """Insere um único embedding."""
        return self.insert([embedding], [person_id], [image_path])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca os embeddings mais similares e converte para dicionário puro.
        Isso corrige o erro 'Hit is not JSON serializable'.
        """
        if output_fields is None:
            output_fields = ["person_id", "image_path", "created_at"]
        
        # Converter numpy para lista se necessário
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_list],
            limit=top_k,
            output_fields=output_fields
        )
        
        # Formatar resultados para JSON serializable
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                # Criar dicionário base com tipos nativos do Python
                item = {
                    "id": hit.id,
                    "distance": float(hit.distance)  # Garante float do Python
                }
                
                # Adicionar campos extras (entity)
                # O objeto Hit possui a propriedade entity que contém os campos solicitados
                entity = getattr(hit, 'entity', {})
                
                # Extrair campos da entidade de forma segura
                for field in output_fields:
                    if hasattr(entity, 'get'):
                        item[field] = entity.get(field)
                
                formatted_results.append(item)
        
        return formatted_results
    
    def query(
        self,
        filter_expr: str = "",
        output_fields: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Consulta registros com filtro genérico."""
        if output_fields is None:
            output_fields = ["id", "person_id", "image_path", "created_at"]
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit
        )
        return results

    def query_by_person(self, person_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Consulta embeddings por person_id.
        Corrige o erro 'MilvusClient object has no attribute query_by_person'.
        """
        return self.query(
            filter_expr=f'person_id == "{person_id}"',
            limit=limit
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da collection."""
        if not self.client.has_collection(self.collection_name):
            return {"row_count": 0}
            
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

    # Alias para compatibilidade com scripts que usam drop_collection
    def drop_collection(self):
        self.delete_collection()

    def recreate_collection(self):
        """Recria a collection (apaga dados existentes)."""
        self.delete_collection()
        self._ensure_collection()