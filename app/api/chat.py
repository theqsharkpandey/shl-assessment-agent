from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.agent import ConversationalAgent

router = APIRouter()
agent = ConversationalAgent()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Standard health check endpoint required by the evaluation harness.
    """
    return HealthResponse(status="ok")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    The stateless conversational endpoint. Receives the full conversation 
    history and passes it to the reasoning agent.
    """
    try:
        # Convert pydantic models to a list of dicts for the LLM pipeline
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Execute the reasoning and retrieval step
        response_data = agent.execute_turn(messages)
        
        return response_data
    except Exception as e:
        import logging
        logging.error(f"Unhandled exception in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
