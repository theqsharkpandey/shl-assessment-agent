import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.chat import router as chat_router

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

app = FastAPI(
    title=settings.APP_NAME,
    description="Stateless Conversational AI Agent for SHL Assessment Recommendations",
    version="2.0.0"
)

# Standard security CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main routers
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    # Typically launched via `uvicorn app.main:app --reload`
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
