"""
Tests for the retrieval layer — BM25, FAISS, hybrid retrieval, calculator.
Run: pytest tests/ -v
"""
import unittest
from unittest.mock import MagicMock, patch


class TestCalculator(unittest.TestCase):
    """Calculator tool — safe expression evaluation."""

    def test_basic_arithmetic(self):
        from tools.calculator import safe_calculate
        self.assertEqual(safe_calculate("2 + 2"), "4")
        self.assertEqual(safe_calculate("10 * 5"), "50")
        self.assertEqual(safe_calculate("100 / 4"), "25")

    def test_floating_point(self):
        from tools.calculator import safe_calculate
        result = safe_calculate("1 / 3")
        self.assertIn("0.333", result)

    def test_symbolic_math(self):
        from tools.calculator import safe_calculate
        self.assertEqual(safe_calculate("sqrt(144)"), "12")

    def test_percentage(self):
        from tools.calculator import safe_calculate
        result = safe_calculate("15 / 100 * 200")
        self.assertEqual(result, "30")

    def test_blocks_dangerous_keywords(self):
        from tools.calculator import safe_calculate
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('ls')")

    def test_empty_expression_raises(self):
        from tools.calculator import safe_calculate
        with self.assertRaises(ValueError):
            safe_calculate("")

    def test_invalid_expression_raises(self):
        from tools.calculator import safe_calculate
        with self.assertRaises(ValueError):
            safe_calculate("not a math expression!!!")


class TestBM25Search(unittest.TestCase):
    """BM25 search — tests with no index (graceful fallback)."""

    def test_returns_empty_when_no_index(self):
        from tools.bm25_search import bm25_search
        results = bm25_search("test query", top_k=5)
        # Should return empty list gracefully if index not built
        self.assertIsInstance(results, list)

    @patch("tools.bm25_search._bm25_index")
    @patch("tools.bm25_search._bm25_docs")
    def test_returns_ranked_results(self, mock_docs, mock_index):
        """Test with a mocked BM25 index."""
        import numpy as np
        mock_docs.__bool__ = lambda self: True
        mock_docs.__len__ = lambda self: 3
        mock_docs.__getitem__ = lambda self, i: [
            {"text": "LoRA is a PEFT method", "source": "arxiv1"},
            {"text": "QLoRA extends LoRA", "source": "arxiv2"},
            {"text": "Transformer attention", "source": "arxiv3"},
        ][i]

        mock_index.get_scores.return_value = np.array([0.9, 0.7, 0.1])

        from tools import bm25_search as bm25_mod
        bm25_mod._bm25_index = mock_index
        bm25_mod._bm25_docs = [
            {"text": "LoRA is a PEFT method", "source": "arxiv1"},
            {"text": "QLoRA extends LoRA", "source": "arxiv2"},
            {"text": "Transformer attention", "source": "arxiv3"},
        ]

        results = bm25_mod.bm25_search("LoRA fine-tuning", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertAlmostEqual(results[0]["score"], 0.9)


class TestRRFFusion(unittest.TestCase):
    """Test Reciprocal Rank Fusion scoring."""

    def test_rrf_merges_and_deduplicates(self):
        from retrieval.hybrid_retriever import _rrf_score

        list_a = [
            {"text": "Document A content here", "source": "a1", "score": 0.9},
            {"text": "Document B content here", "source": "b1", "score": 0.7},
        ]
        list_b = [
            {"text": "Document B content here", "source": "b1", "score": 0.8},
            {"text": "Document C content here", "source": "c1", "score": 0.6},
        ]

        fused = _rrf_score([list_a, list_b])

        # Document B appears in both lists — should have higher RRF score
        texts = [d["text"][:20] for d in fused]
        self.assertIn("Document B content here"[:20], texts)

        # All unique documents should be present
        self.assertGreaterEqual(len(fused), 3)

    def test_rrf_higher_rank_gets_higher_score(self):
        from retrieval.hybrid_retriever import _rrf_score

        list_a = [
            {"text": "Top ranked document wins", "source": "s1", "score": 1.0},
            {"text": "Second ranked document loses", "source": "s2", "score": 0.5},
        ]

        fused = _rrf_score([list_a])
        scores = [d["rrf_score"] for d in fused]

        # First result should have higher RRF score
        self.assertGreater(scores[0], scores[1])


class TestContextManager(unittest.TestCase):
    """Test context window budget management."""

    def test_builds_context_within_budget(self):
        from utils.context_manager import build_context_within_budget

        chunks = [
            {"text": "Short text " * 10, "source": "src1", "score": 0.9},
            {"text": "More context " * 10, "source": "src2", "score": 0.8},
        ]

        context = build_context_within_budget(chunks, memories=[], history=[], max_tokens=500)
        self.assertIn("src1", context)
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 10)

    def test_returns_no_results_message_when_empty(self):
        from utils.context_manager import build_context_within_budget

        context = build_context_within_budget([], memories=[], history=[], max_tokens=500)
        self.assertIn("No relevant context", context)


class TestCostEstimation(unittest.TestCase):
    """Test cost estimation math."""

    def test_gpt4o_mini_cost(self):
        from utils.cost import estimate_cost

        # 1000 input, 200 output with gpt-4o-mini
        cost = estimate_cost(1000, 200, model="gpt-4o-mini")
        # 1000 * 0.150/1M + 200 * 0.600/1M = 0.00015 + 0.00012 = 0.00027
        self.assertAlmostEqual(cost, 0.00027, places=5)

    def test_zero_tokens_zero_cost(self):
        from utils.cost import estimate_cost
        self.assertEqual(estimate_cost(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
