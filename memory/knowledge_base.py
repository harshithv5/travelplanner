import os
from typing import Any
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

_DENSE_NAME   = "dense"
_SPARSE_NAME  = "sparse"
_RRF_K        = 60

# fastembed model names
_DENSE_MODEL  = "BAAI/bge-base-en-v1.5"    # 768-dim
_SPARSE_MODEL = "prithivida/Splade_PP_EN_V1"
_DENSE_DIM    = 768


class KnowledgeBase:
    """
    Qdrant-backed hybrid vector store.

    Embeddings are generated internally — callers pass plain text.
    Dense vectors: BAAI/bge-small-en-v1.5 (384-dim, cosine).
    Sparse vectors: SPLADE (prithivida/Splade_PP_EN_V1).
    Hybrid search fuses both via Reciprocal Rank Fusion.

    Collection name is passed per call so one instance serves many collections.
    """

    def __init__(self, distance: Distance = Distance.COSINE):
        self._client  = QdrantClient(url=os.getenv("QDRANT_URL") , api_key=os.getenv("QDRANT_API_KEY"))
        self._distance = distance
        # Lazy-loaded — populated on first embedding call
        self._dense_model:  TextEmbedding | None        = None
        self._sparse_model: SparseTextEmbedding | None  = None

    # ------------------------------------------------------------------
    # Embedding generation (internal)
    # ------------------------------------------------------------------

    def _load_dense(self) -> TextEmbedding:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(_DENSE_MODEL)
        return self._dense_model

    def _load_sparse(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(_SPARSE_MODEL)
        return self._sparse_model

    def generate_dense_embedding(self, text: str) -> list[float]:
        """
        Encode text into a dense vector using BAAI/bge-small-en-v1.5 (768-dim).

        Args:
            text: Any natural language string.

        Returns:
            List of 384 floats.
        """
        result = list(self._load_dense().embed([text]))[0]
        return result.tolist()

    def generate_sparse_embedding(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode text into a SPLADE sparse vector.

        Args:
            text: Any natural language string.

        Returns:
            (indices, values) — parallel lists of non-zero token positions and weights.
        """
        result = list(self._load_sparse().embed([text]))[0]
        return result.indices.tolist(), result.values.tolist()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        collection: str,
        keyword_index_fields: list[str] | None = None,
    ) -> None:
        """Create the collection with dense + sparse configs if it does not exist.

        Also ensures keyword payload indexes exist for any fields you intend to
        filter on (Qdrant requires an index before filtering on payload keys).
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config={
                    _DENSE_NAME: VectorParams(size=_DENSE_DIM, distance=self._distance)
                },
                sparse_vectors_config={
                    _SPARSE_NAME: SparseVectorParams()
                },
            )

        for field in keyword_index_fields or []:
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(
        self,
        collection: str,
        point_id: str | int,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Embed `text` (dense + sparse) and upsert the point into the collection.

        Args:
            collection: Target collection (created automatically if absent).
            point_id:   Unique identifier (str UUID or int).
            text:       Raw text to embed — no pre-processing needed.
            metadata:   Arbitrary key-value payload stored alongside the vectors.
        """
        self.ensure_collection(collection)

        dense_vec            = self.generate_dense_embedding(text)
        sparse_idx, sparse_w = self.generate_sparse_embedding(text)

        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_NAME:  dense_vec,
                        _SPARSE_NAME: SparseVector(indices=sparse_idx, values=sparse_w),
                    },
                    payload=metadata,
                )
            ],
        )

    def insert_batch(
        self,
        collection: str,
        points: list[dict],
    ) -> None:
        """
        Embed and upsert multiple points in one call.

        Each element of `points` must have:
          - id       : str | int
          - text     : str   — raw text to embed
          - metadata : dict
        """
        self.ensure_collection(collection)

        structs = []
        for p in points:
            dense_vec            = self.generate_dense_embedding(p["text"])
            sparse_idx, sparse_w = self.generate_sparse_embedding(p["text"])
            structs.append(
                PointStruct(
                    id=p["id"],
                    vector={
                        _DENSE_NAME:  dense_vec,
                        _SPARSE_NAME: SparseVector(indices=sparse_idx, values=sparse_w),
                    },
                    payload=p["metadata"],
                )
            )

        self._client.upsert(collection_name=collection, points=structs)

    # ------------------------------------------------------------------
    # Filter-only retrieval (no embedding)
    # ------------------------------------------------------------------

    def scroll(
        self,
        collection: str,
        metadata_filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Filter-only retrieval — no vector search, no embedding computed.

        Use when you just need every point matching a metadata filter (e.g.
        "every hotel in city=shillong") without ranking by similarity.

        Returns:
            List of {id, metadata}.
        """
        qdrant_filter = self._build_filter(metadata_filters) if metadata_filters else None
        points, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [{"id": p.id, "metadata": p.payload} for p in points]

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def dense_search(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Cosine similarity search using the dense (bge-small) vector.

        Args:
            collection:       Collection to query.
            query_text:       Natural language query — embedded internally.
            top_k:            Number of results.
            metadata_filters: Optional {field: value} AND-filters on payload.

        Returns:
            List of {id, score, metadata, source="dense"}.
        """
        qdrant_filter = self._build_filter(metadata_filters) if metadata_filters else None
        query_vec     = self.generate_dense_embedding(query_text)

        response = self._client.query_points(
            collection_name=collection,
            query=query_vec,
            using=_DENSE_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {"id": h.id, "score": h.score, "metadata": h.payload, "source": "dense"}
            for h in response.points
        ]

    # ------------------------------------------------------------------
    # Sparse search
    # ------------------------------------------------------------------

    def sparse_search(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        SPLADE sparse similarity search.

        Args:
            collection:       Collection to query.
            query_text:       Natural language query — SPLADE-encoded internally.
            top_k:            Number of results.
            metadata_filters: Optional {field: value} AND-filters on payload.

        Returns:
            List of {id, score, metadata, source="sparse"}.
        """
        qdrant_filter          = self._build_filter(metadata_filters) if metadata_filters else None
        sparse_idx, sparse_val = self.generate_sparse_embedding(query_text)

        response = self._client.query_points(
            collection_name=collection,
            query=SparseVector(indices=sparse_idx, values=sparse_val),
            using=_SPARSE_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {"id": h.id, "score": h.score, "metadata": h.payload, "source": "sparse"}
            for h in response.points
        ]

    # ------------------------------------------------------------------
    # Hybrid search (dense + sparse → RRF)
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        query_text: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
        rrf_k: int = _RRF_K,
    ) -> list[dict]:
        """
        Hybrid search: SPLADE sparse + bge-small dense, fused via Reciprocal Rank Fusion.

        Duplicates (same ID in both result sets) get RRF contributions from both lists,
        naturally boosting results that match on both signals.

        Args:
            collection:       Collection to query.
            query_text:       Natural language query — both embeddings generated internally.
            top_k:            Final number of results after fusion.
            metadata_filters: Optional {field: value} AND-filters applied to both legs.
            rrf_k:            RRF smoothing constant (default 60).

        Returns:
            List of {id, score, metadata, source} sorted by fused RRF score descending.
            source is "dense", "sparse", or "both".
        """
        fetch_k = top_k * 2

        dense_hits  = self.dense_search(collection,  query_text, fetch_k, metadata_filters)
        sparse_hits = self.sparse_search(collection, query_text, fetch_k, metadata_filters)

        return self._fuse(dense_hits, sparse_hits, top_k=top_k, rrf_k=rrf_k)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(
        dense_hits: list[dict],
        sparse_hits: list[dict],
        top_k: int,
        rrf_k: int,
    ) -> list[dict]:
        """Merge dense and sparse hits via Reciprocal Rank Fusion."""
        rrf_scores: dict[str | int, float] = {}
        registry:   dict[str | int, dict]  = {}

        for rank, hit in enumerate(dense_hits):
            pid = hit["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + 1 + rrf_k)
            registry[pid]   = {"id": pid, "metadata": hit["metadata"], "source": "dense"}

        for rank, hit in enumerate(sparse_hits):
            pid = hit["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + 1 + rrf_k)
            if pid in registry:
                registry[pid]["source"] = "both"
            else:
                registry[pid] = {"id": pid, "metadata": hit["metadata"], "source": "sparse"}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{**registry[pid], "score": round(score, 6)} for pid, score in ranked]

    # ------------------------------------------------------------------
    # Filter builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """
        Build a Qdrant AND-filter from a flat {field: value} dict.

        Args:
            filters: e.g. {"category": "waterfall", "city": "Shillong"}
        """
        return Filter(must=[
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
        ])
