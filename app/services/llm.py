import json
import logging
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """
    Abstract abstraction layer standardizing the interface between
    OpenAI and Google Gemini. This prevents vendor lock-in and satisfies
    clean architecture requirements.
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.MODEL_NAME
        
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, # Determinism is key for strict JSON schemas
                    response_mime_type="application/json"
                )
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _coalesce_gemini_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Gemini's start_chat strictly requires alternating 'user' and 'model' roles.
        This utility coalesces consecutive messages of the same role.
        """
        history = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            if history and history[-1]["role"] == role:
                history[-1]["parts"][0] += "\n\n" + m["content"]
            else:
                history.append({"role": role, "parts": [m["content"]]})
        return history

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """
        Generates a standardized JSON string payload regardless of the backend provider.
        """
        logger.debug(f"Calling {self.provider.upper()} API...")
        
        if self.provider == "openai":
            oai_msgs = [{"role": "system", "content": system_prompt}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=oai_msgs,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
            
        elif self.provider == "gemini":
            # Gemini applies the system prompt via initialization
            model = self.model
            model._system_instruction = {"parts": [{"text": system_prompt}]} # Type mapping abstraction
            
            # Extract history (everything except the last message)
            history_msgs = self._coalesce_gemini_history(messages[:-1])
            last_msg = messages[-1]["content"] if messages else ""
            
            chat = model.start_chat(history=history_msgs)
            response = chat.send_message(last_msg)
            return response.text
