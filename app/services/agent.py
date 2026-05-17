import json
import logging
from typing import List, Dict, Any

from app.services.llm import LLMProvider
from app.retrieval.search import CatalogSearcher
from app.services.prompts import SYSTEM_PROMPT, format_catalog_results
from app.models.schemas import ChatResponse, RecommendationItem

logger = logging.getLogger(__name__)

class ConversationalAgent:
    """
    The orchestration layer that binds the LLM reasoning loop with 
    the FAISS retrieval tools. This runs the state machine.
    """
    def __init__(self):
        self.llm = LLMProvider()
        self.searcher = CatalogSearcher()

    def map_test_type(self, item: dict) -> str:
        """Utility to map categorical keys to legacy test types."""
        keys = item.get("keys", [])
        types = []
        for k in keys:
            k_lower = k.lower()
            if "knowledge" in k_lower and "K" not in types: types.append("K")
            if "personality" in k_lower and "P" not in types: types.append("P")
            if ("ability" in k_lower or "aptitude" in k_lower) and "A" not in types: types.append("A")
            if "simulations" in k_lower and "S" not in types: types.append("S")
            if "competencies" in k_lower and "C" not in types: types.append("C")
            if "biodata" in k_lower and "B" not in types: types.append("B")
            if "development" in k_lower and "D" not in types: types.append("D")
            if "assessment" in k_lower and "E" not in types: types.append("E")
        return ",".join(types) if types else "U"

    def execute_turn(self, messages: List[Dict[str, str]]) -> ChatResponse:
        """
        Executes a single stateless turn. The agent is allowed up to 3 tool calls 
        (internal monologue/retrieval steps) before it MUST yield a response to the user.
        """
        current_messages = list(messages)
        
        # Guardrail: Enforce turn cap by injecting a system note on the 8th user message
        user_message_count = sum(1 for m in current_messages if m["role"] == "user")
        if user_message_count >= 8:
            logger.info("Enforcing 8-turn conversation limit via prompt injection.")
            current_messages[-1]["content"] += "\n\n[SYSTEM NOTE: This is the 8th turn. You MUST complete the recommendation task now and set end_of_conversation to true.]"

        MAX_TOOL_CALLS = 3
        
        for _ in range(MAX_TOOL_CALLS):
            llm_output_str = self.llm.generate(SYSTEM_PROMPT, current_messages)
            
            try:
                output = json.loads(llm_output_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parsing Error from LLM: {e}. Output was: {llm_output_str}")
                # Fallback graceful degradation
                return ChatResponse(
                    reply="I'm sorry, I encountered an internal error. Could you rephrase your request?",
                    recommendations=[],
                    end_of_conversation=False
                )
                
            # If the LLM requests a tool call (Format 1)
            if output.get("type") == "tool_call":
                if output.get("tool") == "search_catalog":
                    query = output.get("query", "")
                    logger.info(f"Agent executing tool: search_catalog with query: '{query}'")
                    
                    results = self.searcher.search(query, top_k=15)
                    context_str = format_catalog_results(results)
                    
                    # Append the tool call and observation to the history so the LLM can "read" it
                    current_messages.append({"role": "assistant", "content": llm_output_str})
                    current_messages.append({"role": "user", "content": json.dumps({"tool_response": context_str})})
                    continue # Loop back and let the LLM reason over the tool response
                else:
                    logger.warning(f"Unknown tool requested: {output.get('tool')}")
                    current_messages.append({"role": "assistant", "content": llm_output_str})
                    current_messages.append({"role": "user", "content": json.dumps({"error": "Unknown tool"})})
                    continue

            # If the LLM yields a final response (Format 2)
            elif output.get("type") == "response":
                reply = output.get("reply", "")
                rec_names = output.get("recommendations", [])
                is_end = output.get("end_of_conversation", False)
                
                final_recs = []
                if rec_names:
                    # Ground the recommendations by fetching precise metadata from the searcher
                    # This prevents hallucinations from reaching the user
                    items = self.searcher.get_by_names(rec_names)
                    for item in items:
                        final_recs.append(RecommendationItem(
                            name=item.get("name", "Unknown"),
                            url=item.get("url", item.get("link", "")),
                            test_type=self.map_test_type(item)
                        ))
                    
                    # Strictly enforce top 10 limit for Recall@10 constraint
                    final_recs = final_recs[:10]
                    
                return ChatResponse(
                    reply=reply,
                    recommendations=final_recs,
                    end_of_conversation=is_end
                )
            
            else:
                logger.error(f"Invalid schema 'type' returned by LLM: {output.get('type')}")
                break
                
        # If the loop exhausts without a response
        return ChatResponse(
            reply="I encountered an error while searching the catalog. Please try again.",
            recommendations=[],
            end_of_conversation=False
        )
