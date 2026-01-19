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
        self.db_path = db_path or Config.MILVUS_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        
        # Conectar ao Milvus Lite
        try:
            self.client = PyMilvusClient(self.db_path)
            print(f"✓ Conectado ao Milvus Lite: {self.db_path}")
        except Exception as e:
            print(f"✗ Erro ao conectar ao Milvus: {e}")
            raise e
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            return
        
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(field_name="person_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=32)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"✓ Collection '{self.collection_name}' criada!")
    
    def insert(self, embeddings: List[np.ndarray], person_ids: List[str], image_paths: List[str]) -> int:
        if not embeddings:
            return 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = []
        for emb, pid, path in zip(embeddings, person_ids, image_paths):
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            data.append({
                "embedding": emb_list,
                "person_id": str(pid),
                "image_path": str(path),
                "created_at": timestamp
            })
        
        self.client.insert(collection_name=self.collection_name, data=data)
        print(f"✓ {len(data)} embeddings inseridos.")
        return len(data)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca os embeddings mais similares e converte para dicionário puro."""
        query = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Realiza a busca
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            limit=top_k,
            output_fields=["person_id", "image_path", "created_at"]
        )
        
        formatted = []
        if results and len(results) > 0:
            for hit in results[0]:
                # Tratamento robusto para 'Hit' object vs Dict
                # Tenta acessar como atributo (objeto), se falhar, tenta como dict
                try:
                    # Tenta pegar entidade (formato objeto)
                    entity = getattr(hit, 'entity', {})
                    # Se entity for vazio, tenta pegar via get (formato dict)
                    if not entity and hasattr(hit, 'get'):
                        entity = hit.get('entity', {})

                    # Extrair valores com fallback
                    f_id = getattr(hit, 'id', hit.get('id') if hasattr(hit, 'get') else None)
                    f_dist = getattr(hit, 'distance', hit.get('distance') if hasattr(hit, 'get') else 0.0)
                    
                    # Garantir que entity é dict
                    if not isinstance(entity, dict):
                         # Fallback forçado se entity não for dict acessível
                         entity = {}

                    formatted.append({
                        "id": f_id,
                        "distance": float(f_dist), # Garantir float nativo do Python
                        "person_id": entity.get("person_id", "Unknown"),
                        "image_path": entity.get("image_path", ""),
                        "created_at": entity.get("created_at", "")
                    })
                except Exception as e:
                    print(f"Erro ao processar hit: {e}")
                    continue
                    
        return formatted

    def query_by_person(self, person_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'person_id == "{person_id}"',
            output_fields=["id", "person_id", "image_path", "created_at"],
            limit=limit
        )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.client.has_collection(self.collection_name):
            return {"row_count": 0, "exists": False}
        stats = self.client.get_collection_stats(self.collection_name)
        return {"exists": True, "row_count": stats.get("row_count", 0)}

    def list_collections(self) -> List[str]:
        return self.client.list_collections()