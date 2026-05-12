"""
app/milvus_client.py
=====================
Cliente para operações no banco de dados vetorial Milvus Lite.
Suporta embeddings de 1024 dimensões (com TTA).

Mudança em relação à versão anterior:
  - Normalização L2 aplicada em insert() e search() antes de qualquer
    operação no banco, garantindo que todos os vetores estejam na
    mesma escala unitária independente da origem.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from pymilvus import MilvusClient as PyMilvusClient, DataType

from .config import Config


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Normaliza um embedding para norma unitária (L2).
    Operação idempotente: se já normalizado, não altera.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return embedding
    return embedding / norm


class MilvusClient:
    """Cliente para operações no Milvus Lite."""

    def __init__(self, db_path: str = None, collection_name: str = None):
        self.db_path = db_path or Config.MILVUS_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM

        try:
            self.client = PyMilvusClient(self.db_path)
            print(f"✓ Conectado ao Milvus Lite: {self.db_path}")
            print(f"  Embedding dim: {self.embedding_dim}")
        except Exception as e:
            print(f"✗ Erro ao conectar no Milvus: {e}")
            raise e

        self._ensure_collection()

    def _ensure_collection(self):
        """Garante que a collection existe com o schema correto."""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field(field_name="id",         datatype=DataType.INT64,         is_primary=True)
        schema.add_field(field_name="embedding",  datatype=DataType.FLOAT_VECTOR,  dim=self.embedding_dim)
        schema.add_field(field_name="person_id",  datatype=DataType.VARCHAR,       max_length=256)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR,       max_length=512)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR,       max_length=32)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"✓ Collection '{self.collection_name}' criada com dim={self.embedding_dim}!")

    # -----------------------------------------------------------------------
    # Insert
    # -----------------------------------------------------------------------
    def insert(
        self,
        embeddings: List[np.ndarray],
        person_ids: List[str],
        image_paths: List[str],
    ) -> int:
        """
        Insere embeddings no Milvus.
        Aplica normalização L2 antes de inserir para garantir consistência.
        """
        if not embeddings:
            return 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = []
        for emb, pid, path in zip(embeddings, person_ids, image_paths):
            emb_array = np.array(emb) if not isinstance(emb, np.ndarray) else emb

            # Normalização L2 antes de inserir
            emb_normalized = _l2_normalize(emb_array)

            data.append({
                "embedding": emb_normalized.tolist(),
                "person_id": str(pid),
                "image_path": str(path),
                "created_at": timestamp,
            })

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        return len(data)

    def insert_single(self, embedding: np.ndarray, person_id: str, image_path: str) -> int:
        """Insere um único embedding."""
        return self.insert([embedding], [person_id], [image_path])

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        output_fields: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca os embeddings mais similares.
        Aplica normalização L2 no vetor de query antes de buscar.
        """
        if output_fields is None:
            output_fields = ["person_id", "image_path", "created_at"]

        # Normalização L2 antes de buscar
        query_array = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding
        query_normalized = _l2_normalize(query_array)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_normalized.tolist()],
            limit=top_k,
            output_fields=output_fields,
        )

        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                item = {
                    "id": hit.id,
                    "distance": float(hit.distance),
                }
                entity = getattr(hit, "entity", {})
                for field in output_fields:
                    if hasattr(entity, "get"):
                        item[field] = entity.get(field)
                formatted_results.append(item)

        return formatted_results

    # -----------------------------------------------------------------------
    # Query / Stats (sem alteração)
    # -----------------------------------------------------------------------
    def query(
        self,
        filter_expr: str = "",
        output_fields: List[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Consulta registros com filtro genérico."""
        if output_fields is None:
            output_fields = ["id", "person_id", "image_path", "created_at"]

        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

    def query_by_person(self, person_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Consulta embeddings por person_id."""
        return self.query(
            filter_expr=f'person_id == "{person_id}"',
            limit=limit,
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da collection."""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            count = int(stats.get("row_count", 0))
        except Exception:
            count = 0

        return {
            "collection": self.collection_name,
            "embedding_dim": self.embedding_dim,
            "total_embeddings": count,
            "db_path": self.db_path,
        }

    def drop_collection(self) -> bool:
        """Remove a collection."""
        try:
            self.client.drop_collection(self.collection_name)
            return True
        except Exception:
            return False

    def has_collection(self) -> bool:
        """Verifica se a collection existe."""
        return self.client.has_collection(self.collection_name)

    def recreate_collection(self) -> None:
        """Remove e recria a collection do zero."""
        if self.has_collection():
            self.drop_collection()
            print(f"✓ Collection '{self.collection_name}' removida.")
        self._ensure_collection()
        print(f"✓ Collection '{self.collection_name}' recriada.")

    @staticmethod
    def purge_model_collections(model_name: str, db_path: Optional[str] = None) -> List[str]:
        """
        Politica "uma collection por modelo": dropa TODAS as collections
        relacionadas a um modelo antes de uma nova populacao.

        Inclui:
          - O nome canonico (Config.get_collection_name(model_name))
          - Qualquer variante com sufixo (ex: nome_canonico + "_lfw_ext_val",
            "_ext_val", etc) — herancas de versoes anteriores quando o codigo
            permitia multiplas collections por modelo.

        Retorna a lista de collections dropadas (vazia se nada existia).
        """
        canonical = Config.get_collection_name(model_name)
        actual_db = db_path or Config.MILVUS_DB_PATH

        client = PyMilvusClient(actual_db)
        try:
            existing = client.list_collections() or []
        except Exception:
            existing = []

        # Dropa o canonico + qualquer variante "canonical_<sufixo>"
        to_drop = [
            c for c in existing
            if c == canonical or c.startswith(canonical + "_")
        ]

        for coll in to_drop:
            try:
                client.drop_collection(coll)
                print(f"  ✓ Collection removida: {coll}")
            except Exception as e:
                print(f"  ⚠ Falha ao dropar '{coll}': {e}")

        return to_drop