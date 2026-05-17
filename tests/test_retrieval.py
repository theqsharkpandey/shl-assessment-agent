import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.search import CatalogSearcher

@pytest.fixture
def mock_searcher():
    with patch("app.retrieval.search.faiss") as mock_faiss, \
         patch("app.retrieval.search.SentenceTransformer") as mock_st, \
         patch("app.retrieval.search.os.path.exists", return_value=True), \
         patch("builtins.open", new_callable=MagicMock):
        
        searcher = CatalogSearcher()
        searcher.is_ready = True
        searcher.metadata = [
            {"name": "Java 8 Assessment", "keys": ["Knowledge"]},
            {"name": "OPQ32", "keys": ["Personality"]}
        ]
        return searcher

def test_get_by_names(mock_searcher):
    """Ensure case-insensitive matching works effectively."""
    # Exact match
    results = mock_searcher.get_by_names(["Java 8 Assessment"])
    assert len(results) == 1
    
    # Case mismatch and trailing space
    results = mock_searcher.get_by_names(["  java 8 assessment  "])
    assert len(results) == 1
    
    # Not found
    results = mock_searcher.get_by_names(["Unknown Assessment"])
    assert len(results) == 0
