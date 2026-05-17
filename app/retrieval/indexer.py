import os
import json
import faiss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CatalogIndexer:
    """
    A robust indexing pipeline using FAISS and SentenceTransformers.
    It reads the cleaned JSON from the scraper, chunks text if necessary,
    generates dense embeddings, and serializes the index to disk.
    """
    
    def __init__(self, 
                 raw_data_path: str = "data/raw_catalog.json",
                 index_dir: str = "data/index",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.raw_data_path = raw_data_path
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "catalog.faiss")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        
        # Using a fast, lightweight sentence-transformer model ideal for semantic search
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        os.makedirs(self.index_dir, exist_ok=True)

    def load_data(self):
        """Loads cleaned catalog data."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}. Run scraper.py first.")
            
        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_document_representation(self, item: dict) -> str:
        """
        Creates a dense textual representation of the item to be embedded.
        We inject categorical keys and descriptions into the semantic space.
        """
        name = item.get("name", "")
        desc = item.get("description", "")
        keys = ", ".join(item.get("keys", []))
        
        # This context engineering provides maximum surface area for semantic matching
        return f"Assessment Name: {name}\nCategories: {keys}\nDescription: {desc}"

    def run(self):
        logger.info("Starting indexing pipeline...")
        items = self.load_data()
        
        if not items:
            logger.error("No items to index.")
            return

        # Prepare documents and metadata
        documents = []
        metadata = []
        
        for idx, item in enumerate(items):
            doc_text = self.build_document_representation(item)
            documents.append(doc_text)
            
            # Store full item for retrieval context, plus the internal FAISS id
            item["_faiss_id"] = idx
            metadata.append(item)

        logger.info(f"Generating embeddings for {len(documents)} assessments...")
        # Encode all documents (returns numpy array)
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Ensure correct dtype for FAISS
        embeddings = np.array(embeddings).astype("float32")
        
        # Determine embedding dimension
        dim = embeddings.shape[1]
        
        # Initialize FAISS Index (L2 distance is standard for unnormalized vectors, 
        # but InnerProduct is better for cosine similarity if we normalize).
        # We will normalize and use Inner Product for Cosine Similarity.
        logger.info("Building FAISS index (Cosine Similarity)...")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        
        # Add vectors to index
        index.add(embeddings)
        
        logger.info(f"Index built with {index.ntotal} vectors.")
        
        # Serialize to disk
        faiss.write_index(index, self.index_path)
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Index saved to {self.index_path}")
        logger.info(f"Metadata saved to {self.metadata_path}")

if __name__ == "__main__":
    indexer = CatalogIndexer()
    indexer.run()
