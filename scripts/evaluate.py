import json
import logging
from typing import List

from app.services.agent import ConversationalAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluation framework to run programmatic assertions on the agent
    to calculate Recall@10 and run Behavioral Probes.
    """
    def __init__(self):
        self.agent = ConversationalAgent()

    def test_behavioral_probe_vague_query(self) -> bool:
        """
        Probe: Agent must not recommend anything on Turn 1 if the query is vague.
        """
        messages = [{"role": "user", "content": "I need an assessment."}]
        response = self.agent.execute_turn(messages)
        
        # Assertion 1: Recommendations must be empty
        passed = len(response.recommendations) == 0
        
        # Assertion 2: Must not end conversation
        passed = passed and (response.end_of_conversation is False)
        
        logger.info(f"Behavioral Probe (Vague Query): {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_behavioral_probe_off_topic(self) -> bool:
        """
        Probe: Agent must gracefully refuse legal or general hiring policy advice.
        """
        messages = [{"role": "user", "content": "What is the legal risk of firing someone in California?"}]
        response = self.agent.execute_turn(messages)
        
        passed = len(response.recommendations) == 0
        logger.info(f"Behavioral Probe (Off Topic Refusal): {'PASSED' if passed else 'FAILED'}")
        return passed

    def compute_recall_at_k(self, query: str, expected_names: List[str], k: int = 10) -> float:
        """
        Calculates Recall@K for a synthetic test query.
        Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments for the query)
        """
        # We manually use the searcher to simulate what the LLM will see
        results = self.agent.searcher.search(query, top_k=k)
        retrieved_names = [r.get("name") for r in results]
        
        hits = sum(1 for expected in expected_names if expected in retrieved_names)
        recall = hits / len(expected_names) if expected_names else 0.0
        
        logger.info(f"Recall@{k} for '{query}': {recall:.2f}")
        return recall

    def run_all(self):
        logger.info("Starting Evaluation Suite...")
        
        # Run probes
        self.test_behavioral_probe_vague_query()
        self.test_behavioral_probe_off_topic()
        
        # Mock Recall calculation
        # This requires known test fixtures for ground truth
        expected_dev_tests = [".NET MVC (New)", "Java 8 (New)", "Spring Framework (New)"]
        self.compute_recall_at_k("Backend Java and .NET framework assessments", expected_dev_tests)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_all()
