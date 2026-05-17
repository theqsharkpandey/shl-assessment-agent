import os
import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

class CatalogSearcher:
    """
    Production-ready FAISS retrieval class.
    Handles loading the serialized index and providing fast vector similarity search.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern to prevent reloading massive models and indices
        if cls._instance is None:
            cls._instance = super(CatalogSearcher, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.index_path = settings.FAISS_INDEX_PATH
        self.metadata_path = settings.CATALOG_METADATA_PATH
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.warning("FAISS index not found. Agent will lack grounding until indexer.py is run.")
            self.is_ready = False
            return

        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)
        
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.is_ready = True

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Executes a dense vector search across the SHL catalog.
        """
        if not self.is_ready or not query:
            return []
            
        # Generate query embedding
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb).astype("float32")
        faiss.normalize_L2(query_emb)
        
        # Search index
        # D = Distances (cosine similarity since we normalized L2 and use Inner Product)
        # I = Indices
        D, I = self.index.search(query_emb, top_k)
        
        results = []
        for i, dist in zip(I[0], D[0]):
            if i != -1: # FAISS returns -1 if there aren't enough neighbors
                item = self.metadata[i]
                # We inject the similarity score to allow thresholding or LLM reranking insights
                item["_score"] = float(dist)
                results.append(item)
                
        return results

    def get_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Robust fetching mechanism for final recommendations.
        Uses case-insensitive and stripped matching to prevent LLM hallucination drops.
        """
        if not self.is_ready:
            return []
            
        results = []
        for name in names:
            target = name.strip().lower()
            for item in self.metadata:
                if item.get("name", "").strip().lower() == target:
                    results.append(item)
                    break # Break inner loop on match
        return results
