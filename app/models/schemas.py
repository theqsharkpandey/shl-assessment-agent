from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class Message(BaseModel):
    role: str = Field(..., description="Role of the sender (user, assistant, or system)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, description="List of previous conversation turns")

class RecommendationItem(BaseModel):
    name: str = Field(..., description="Exact name of the assessment")
    url: str = Field(..., description="Valid URL to the assessment page")
    test_type: str = Field(..., description="Letter mapping for the test type (e.g. K, P, A)")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The conversational response from the agent")
    recommendations: List[RecommendationItem] = Field(
        default_factory=list, 
        description="A list of 1-10 recommended assessments, or empty if clarifying"
    )
    end_of_conversation: bool = Field(..., description="True if the interaction is concluded")

class HealthResponse(BaseModel):
    status: str = Field(default="ok")
