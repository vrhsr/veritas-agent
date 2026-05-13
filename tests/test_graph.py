"""
Integration test for the full LangGraph pipeline.
Uses mock LLMs to run the entire graph without API calls.
Run: pytest tests/test_graph.py -v
"""
import json
import unittest
from unittest.mock import MagicMock, patch


def _mock_llm_response(content: str):
    """Create a mock ChatOpenAI response."""
    mock = MagicMock()
    mock.content = content
    return mock


QUERY_ANALYZER_RESPONSE = json.dumps({
    "query_type": "complex",
    "named_entities": ["LoRA", "QLoRA"],
    "required_tools": ["vector_search", "bm25_search"],
    "complexity_score": 0.8,
})

REASONING_RESPONSE = json.dumps({
    "sub_questions": ["What is LoRA?", "How does it reduce memory?"],
    "sub_answers": ["LoRA uses low-rank adapters.", "Only small matrices are trained."],
    "gaps": [],
    "final_answer": "LoRA reduces memory by training only low-rank adapter matrices instead of all weights.",
    "confidence": 0.85,
})

VALIDATION_RESPONSE = json.dumps({
    "passed": True,
    "grounding": True,
    "consistency": True,
    "completeness": True,
    "failures": [],
})

RESPONSE_GEN_RESPONSE = (
    "LoRA (Low-Rank Adaptation) reduces GPU memory requirements by adding small, "
    "trainable low-rank matrices to frozen pre-trained weights [1]. "
    "Instead of updating all model parameters, it only trains these adapter matrices, "
    "which can reduce trainable parameters by 10,000x [2].\n\n"
    "**Sources:**\n[1] arxiv.org/abs/2106.09685\n[2] arxiv.org/wiki"
)


class TestFullGraphIntegration(unittest.TestCase):
    """Integration test: full graph run with all LLM calls mocked."""

    @patch("agent.nodes.query_analyzer.load_prompt", return_value="mock system prompt")
    @patch("agent.nodes.reasoning_agent.load_prompt", return_value="mock system prompt")
    @patch("agent.nodes.validation_agent.load_prompt", return_value="mock system prompt")
    @patch("agent.nodes.response_generator.load_prompt", return_value="mock system prompt")
    @patch("agent.nodes.query_analyzer.ChatOpenAI")
    @patch("agent.nodes.reasoning_agent.ChatOpenAI")
    @patch("agent.nodes.validation_agent.ChatOpenAI")
    @patch("agent.nodes.response_generator.ChatOpenAI")
    @patch("agent.nodes.memory_manager.RedisMemory")
    @patch("agent.nodes.memory_manager.FAISSMemory")
    @patch("agent.nodes.retrieval_agent.hybrid_retrieve")
    def test_complex_query_full_pipeline(
        self,
        mock_retrieve,
        mock_faiss_cls,
        mock_redis_cls,
        mock_resp_llm_cls,
        mock_val_llm_cls,
        mock_reason_llm_cls,
        mock_qa_llm_cls,
        *mock_prompts,
    ):
        """Test that a complex query traverses the full pipeline correctly."""
        # Set up mock retrieval
        mock_retrieve.return_value = [
            {"text": "LoRA paper content about low-rank adaptation.", "source": "arxiv:2106.09685", "score": 0.9},
            {"text": "Memory comparison between LoRA and full fine-tuning.", "source": "arxiv:2305", "score": 0.8},
        ]

        # Set up mock LLMs
        mock_qa_llm_cls.return_value.invoke.return_value = _mock_llm_response(QUERY_ANALYZER_RESPONSE)
        mock_reason_llm_cls.return_value.invoke.return_value = _mock_llm_response(REASONING_RESPONSE)
        mock_val_llm_cls.return_value.invoke.return_value = _mock_llm_response(VALIDATION_RESPONSE)
        mock_resp_llm_cls.return_value.invoke.return_value = _mock_llm_response(RESPONSE_GEN_RESPONSE)

        # Set up mock memory
        mock_redis = MagicMock()
        mock_redis.get_history.return_value = []
        mock_redis_cls.return_value = mock_redis
        mock_faiss = MagicMock()
        mock_faiss.search_memories.return_value = []
        mock_faiss.check_semantic_cache.return_value = None
        mock_faiss_cls.return_value = mock_faiss

        from agent.graph import run_query
        result = run_query("How does LoRA reduce memory compared to full fine-tuning?", session_id="test-123")

        # Assertions
        self.assertEqual(result["query_type"], "complex")
        self.assertGreater(result["confidence"], 0.7)
        self.assertTrue(result["validation_passed"])
        self.assertIn("LoRA", result["final_answer"])
        self.assertGreater(result["cost_usd"], 0.0)
        self.assertIn("query_analyzer", result["node_latencies"])
        self.assertIn("retrieval_agent", result["node_latencies"])
        self.assertIn("reasoning_agent", result["node_latencies"])
        self.assertIn("validation_agent", result["node_latencies"])
        self.assertIn("response_generator", result["node_latencies"])

    @patch("agent.nodes.query_analyzer.load_prompt", return_value="mock prompt")
    @patch("agent.nodes.response_generator.load_prompt", return_value="mock prompt")
    @patch("agent.nodes.query_analyzer.ChatOpenAI")
    @patch("agent.nodes.response_generator.ChatOpenAI")
    @patch("agent.nodes.memory_manager.RedisMemory")
    @patch("agent.nodes.memory_manager.FAISSMemory")
    def test_simple_query_skips_retrieval(
        self, mock_faiss_cls, mock_redis_cls,
        mock_resp_llm_cls, mock_qa_llm_cls, *_
    ):
        """Simple queries should skip retrieval and go directly to response."""
        simple_qa_response = json.dumps({
            "query_type": "simple",
            "named_entities": ["BERT"],
            "required_tools": [],
            "complexity_score": 0.2,
        })
        mock_qa_llm_cls.return_value.invoke.return_value = _mock_llm_response(simple_qa_response)
        mock_resp_llm_cls.return_value.invoke.return_value = _mock_llm_response(
            "BERT is a bidirectional encoder representation from transformers."
        )

        mock_redis = MagicMock()
        mock_redis.get_history.return_value = []
        mock_redis_cls.return_value = mock_redis
        mock_faiss = MagicMock()
        mock_faiss.search_memories.return_value = []
        mock_faiss_cls.return_value = mock_faiss

        from agent.graph import run_query
        result = run_query("What is BERT?", session_id="test-simple")

        self.assertEqual(result["query_type"], "simple")
        # No retrieval — chunks should be empty
        self.assertEqual(result.get("retrieved_chunks", []), [])
        # Should still have a final answer
        self.assertGreater(len(result.get("final_answer", "")), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
