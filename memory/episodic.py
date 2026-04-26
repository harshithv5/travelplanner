import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION = "episodic_memory"


class EpisodicMemory:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def store(self, trip_id: str, summary: str, vector: list):
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"trip_id": trip_id, "summary": summary},
                )
            ],
        )

    def search(self, query_vector: list, top_k: int = 3) -> list:
        results = self.client.search(
            collection_name=COLLECTION, query_vector=query_vector, limit=top_k
        )
        return [r.payload for r in results]
