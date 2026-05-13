"""
Tests for individual agent nodes.
Uses unittest.mock to avoid real LLM calls.
Run: pytest tests/ -v
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from agent.state import AgentState


def _make_state(**overrides) -> AgentState:
    """Create a minimal valid AgentState for testing."""
    base = AgentState(
        original_query="How does LoRA reduce memory?",
        session_id="test-session",
        query_type="",
        named_entities=[],
        required_tools=[],
        complexity_score=0.0,
        clarification_question="",
        clarification_answer="",
        awaiting_clarification=False,
        retrieved_chunks=[],
        retrieval_strategy="",
        sub_questions=[],
        sub_answers=[],
        gaps=[],
        reasoning_output="",
        confidence=0.0,
        validation_passed=False,
        validation_failures=[],
        retry_count=0,
        final_answer="",
        cited_sources=[],
        relevant_memories=[],
        cost_inr=0.0,
        node_latencies={},
        token_usage={},
        error_log=[],
        prompt_version="v1",
    )
    base.update(overrides)
    return base


class TestQueryAnalyzerRouting(unittest.TestCase):
    """Test routing logic without LLM calls."""

    def test_route_simple(self):
        from agent.nodes.query_analyzer import route_after_query_analyzer
        state = _make_state(query_type="simple")
        result = route_after_query_analyzer(state)
        self.assertEqual(result, "response_generator")

    def test_route_complex(self):
        from agent.nodes.query_analyzer import route_after_query_analyzer
        state = _make_state(query_type="complex")
        result = route_after_query_analyzer(state)
        self.assertEqual(result, "retrieval_agent")

    def test_route_ambiguous(self):
        from agent.nodes.query_analyzer import route_after_query_analyzer
        state = _make_state(query_type="ambiguous")
        result = route_after_query_analyzer(state)
        self.assertEqual(result, "clarification_agent")

    def test_route_defaults_to_retrieval(self):
        from agent.nodes.query_analyzer import route_after_query_analyzer
        state = _make_state(query_type="")
        result = route_after_query_analyzer(state)
        self.assertEqual(result, "retrieval_agent")


class TestReasoningRouting(unittest.TestCase):
    """Test confidence-based routing logic."""

    def test_high_confidence_passes(self):
        from agent.nodes.reasoning_agent import route_after_reasoning
        state = _make_state(confidence=0.85, retry_count=0)
        result = route_after_reasoning(state)
        self.assertEqual(result, "validation_agent")

    def test_border_confidence_retries(self):
        from agent.nodes.reasoning_agent import route_after_reasoning
        state = _make_state(confidence=0.60, retry_count=0)
        result = route_after_reasoning(state)
        self.assertEqual(result, "retrieval_agent")

    def test_border_confidence_max_retries_clarifies(self):
        from agent.nodes.reasoning_agent import route_after_reasoning
        state = _make_state(confidence=0.60, retry_count=2)
        result = route_after_reasoning(state)
        self.assertEqual(result, "clarification_agent")

    def test_low_confidence_clarifies(self):
        from agent.nodes.reasoning_agent import route_after_reasoning
        state = _make_state(confidence=0.30, retry_count=0)
        result = route_after_reasoning(state)
        self.assertEqual(result, "clarification_agent")


class TestValidationRouting(unittest.TestCase):
    """Test validation pass/fail routing."""

    def test_validation_pass_routes_to_response(self):
        from agent.nodes.validation_agent import route_after_validation
        state = _make_state(validation_passed=True, retry_count=0)
        result = route_after_validation(state)
        self.assertEqual(result, "response_generator")

    def test_validation_fail_retries(self):
        from agent.nodes.validation_agent import route_after_validation
        state = _make_state(validation_passed=False, retry_count=1)
        result = route_after_validation(state)
        self.assertEqual(result, "retrieval_agent")

    def test_validation_fail_max_retries_responds(self):
        from agent.nodes.validation_agent import route_after_validation
        state = _make_state(validation_passed=False, retry_count=2)
        result = route_after_validation(state)
        self.assertEqual(result, "response_generator")


class TestQueryAnalyzerNode(unittest.TestCase):
    """Test query_analyzer_node with mocked LLM."""

    @patch("agent.nodes.query_analyzer.ChatOpenAI")
    @patch("agent.nodes.query_analyzer.load_prompt", return_value="test prompt")
    def test_analyzes_complex_query(self, mock_prompt, mock_llm_cls):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "complex",
            "named_entities": ["LoRA"],
            "required_tools": ["vector_search", "bm25_search"],
            "complexity_score": 0.8,
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from agent.nodes.query_analyzer import query_analyzer_node
        state = _make_state()
        result = query_analyzer_node(state)

        self.assertEqual(result["query_type"], "complex")
        self.assertIn("LoRA", result["named_entities"])
        self.assertAlmostEqual(result["complexity_score"], 0.8)

    @patch("agent.nodes.query_analyzer.ChatOpenAI")
    @patch("agent.nodes.query_analyzer.load_prompt", return_value="test prompt")
    def test_handles_json_parse_error(self, mock_prompt, mock_llm_cls):
        mock_response = MagicMock()
        mock_response.content = "NOT VALID JSON {{{"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from agent.nodes.query_analyzer import query_analyzer_node
        state = _make_state()
        result = query_analyzer_node(state)

        # Should fall back to complex
        self.assertEqual(result["query_type"], "complex")


class TestReasoningNode(unittest.TestCase):
    """Test reasoning_agent_node with mocked LLM."""

    @patch("agent.nodes.reasoning_agent.ChatOpenAI")
    @patch("agent.nodes.reasoning_agent.load_prompt", return_value="test prompt")
    def test_extracts_confidence(self, mock_prompt, mock_llm_cls):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sub_questions": ["What is LoRA?", "How does it save memory?"],
            "sub_answers": ["LoRA adds low-rank adapters.", "It only trains small matrices."],
            "gaps": [],
            "final_answer": "LoRA reduces memory by training only low-rank adapter matrices.",
            "confidence": 0.82,
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from agent.nodes.reasoning_agent import reasoning_agent_node
        state = _make_state(
            retrieved_chunks=[{"text": "LoRA paper content", "source": "arxiv", "score": 0.9}]
        )
        result = reasoning_agent_node(state)

        self.assertAlmostEqual(result["confidence"], 0.82)
        self.assertEqual(len(result["sub_questions"]), 2)
        self.assertIn("LoRA", result["reasoning_output"])


class TestValidationNode(unittest.TestCase):
    """Test validation_agent_node with mocked LLM."""

    @patch("agent.nodes.validation_agent.ChatOpenAI")
    @patch("agent.nodes.validation_agent.load_prompt", return_value="test prompt")
    def test_passes_valid_answer(self, mock_prompt, mock_llm_cls):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "passed": True,
            "grounding": True,
            "consistency": True,
            "completeness": True,
            "failures": [],
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from agent.nodes.validation_agent import validation_agent_node
        state = _make_state(
            reasoning_output="LoRA reduces memory by training adapters.",
            retrieved_chunks=[{"text": "LoRA content", "source": "arxiv", "score": 0.9}],
        )
        result = validation_agent_node(state)

        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["validation_failures"], [])

    @patch("agent.nodes.validation_agent.ChatOpenAI")
    @patch("agent.nodes.validation_agent.load_prompt", return_value="test prompt")
    def test_increments_retry_on_fail(self, mock_prompt, mock_llm_cls):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "passed": False,
            "grounding": False,
            "consistency": True,
            "completeness": True,
            "failures": ["Claim X not found in context"],
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        from agent.nodes.validation_agent import validation_agent_node
        state = _make_state(retry_count=0)
        result = validation_agent_node(state)

        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["retry_count"], 1)  # Incremented


if __name__ == "__main__":
    unittest.main(verbosity=2)
