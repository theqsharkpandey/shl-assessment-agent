import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings, automatically populated from environment variables
    or a .env file if present.
    """
    APP_NAME: str = "SHL Conversational Agent"
    APP_ENV: str = "development"
    
    # LLM Settings
    LLM_PROVIDER: str = "gemini" # Options: 'openai', 'gemini'
    MODEL_NAME: str = "gemini-1.5-pro"
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # RAG Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/index/catalog.faiss"
    CATALOG_METADATA_PATH: str = "data/index/metadata.json"
    RAW_CATALOG_PATH: str = "data/raw_catalog.json"
    
    # Scraping Config
    SHL_CATALOG_URL: str = "https://www.shl.com/solutions/products/product-catalog/"

    class Config:
        env_file = ".env"
        case_sensitive = True

# Instantiate a global settings object
settings = Settings()

# Validation block
if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
    # We don't raise error at import time to allow tests/builds to run,
    # but in a strict prod environment, we might want to fail fast here.
    pass
