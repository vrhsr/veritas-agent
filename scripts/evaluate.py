"""
evaluate.py — RAGAS evaluation pipeline on test queries.

Run: python scripts/evaluate.py [--max-queries N]
"""
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import time
from typing import List

from datasets import Dataset
from ragas import evaluate
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
except ImportError:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
from tqdm import tqdm

from agent.graph import run_query

EVAL_DIR = Path("data/eval")
RESULTS_DIR = Path("data/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_QUERIES_FILE = EVAL_DIR / "test_queries.jsonl"

SAMPLE_QUERIES = [
    # Simple — direct lookup from internal docs
    {"query": "Which payment processor does Meesho use?", "expected_query_type": "simple"},
    {"query": "What is Acme Corp's annual GMV?", "expected_query_type": "simple"},
    {"query": "Who is the account manager for CRED?", "expected_query_type": "simple"},
    {"query": "What is the standard Razorpay MDR for credit cards?", "expected_query_type": "simple"},

    # Complex — multi-hop, needs retrieval + reasoning
    {"query": "Compare Razorpay and Stripe international transaction fees and tell me which one is cheaper for a business doing 60% international revenue.", "expected_query_type": "complex"},
    {"query": "Which of our enterprise clients are currently using Stripe and what is their combined annual processing volume?", "expected_query_type": "complex"},
    {"query": "Which clients are at high churn risk and what are the specific reasons for each?", "expected_query_type": "complex"},
    {"query": "Dunzo is on Stripe paying 2.9% MDR. How much would they save annually if they switched to Razorpay at 1.85%?", "expected_query_type": "complex"},
    {"query": "Which clients have had unresolved support escalations in 2024 and what is the current status?", "expected_query_type": "complex"},
    {"query": "For clients processing above 500 crore annually, what negotiated MDR rates do they currently have with Razorpay?", "expected_query_type": "complex"},
    {"query": "Should BrowserStack use Razorpay or Stripe given that 78% of their revenue is international?", "expected_query_type": "complex"},
    {"query": "What is the effective MDR for Zepto after accounting for their UPI transaction mix and GST input credit?", "expected_query_type": "complex"},
    {"query": "Which clients would benefit most from adding BNPL based on their average ticket size and current processor?", "expected_query_type": "complex"},
    {"query": "Compare Razorpay and Stripe on settlement timelines, chargeback fees, and webhook reliability for an enterprise client.", "expected_query_type": "complex"},

    # Ambiguous — needs clarification
    {"query": "Which clients should we focus on?", "expected_query_type": "ambiguous"},
    {"query": "What are the fees?", "expected_query_type": "ambiguous"},
    {"query": "Is Stripe better?", "expected_query_type": "ambiguous"},

    # Multi-source — needs internal docs + web search
    {"query": "Juspay is complaining about settlement delays. What does our internal record say about their issue and what is Razorpay's current SLA for enterprise settlement?", "expected_query_type": "complex"},
    {"query": "Groww's NACH rejection rate is 6%. What is the industry benchmark and what fix has been proposed?", "expected_query_type": "complex"},
]


def load_test_queries() -> List[dict]:
    if not TEST_QUERIES_FILE.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(TEST_QUERIES_FILE, "w", encoding="utf-8") as f:
            for q in SAMPLE_QUERIES:
                f.write(json.dumps(q) + "\n")
        print(f"Created {len(SAMPLE_QUERIES)} sample queries -> {TEST_QUERIES_FILE}")

    queries = []
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def run_evaluation(max_queries: int = 20) -> dict:
    print("=" * 60)
    print("RAGAS Evaluation Pipeline")
    print("=" * 60)
    test_queries = load_test_queries()[:max_queries]
    print(f"Evaluating {len(test_queries)} queries...\n")

    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    run_records = []
    latencies = []
    completed = 0
    routing_correct = 0

    for item in tqdm(test_queries, desc="Running agent"):
        query = item["query"]
        expected_type = item.get("expected_query_type", "")
        start = time.perf_counter()
        try:
            result = run_query(query, session_id=f"eval-{hash(query)}")
            elapsed = time.perf_counter() - start

            if result.get("awaiting_clarification"):
                if expected_type == "ambiguous":
                    routing_correct += 1
                continue

            completed += 1
            latencies.append(elapsed)
            if expected_type and result.get("query_type") == expected_type:
                routing_correct += 1

            ragas_data["question"].append(query)
            ragas_data["answer"].append(result.get("final_answer", ""))
            ragas_data["contexts"].append([c.get("text", "") for c in result.get("retrieved_chunks", [])[:5]])
            ragas_data["ground_truth"].append(item.get("ground_truth", ""))
            run_records.append({
                "query": query,
                "query_type": result.get("query_type"),
                "confidence": result.get("confidence", 0.0),
                "validation_passed": result.get("validation_passed", False),
                "retry_count": result.get("retry_count", 0),
                "latency_s": round(elapsed, 3),
                "cost_inr": result.get("cost_inr", 0.0),
            })
        except Exception as e:
            print(f"  ERROR: {query[:50]}: {e}")

    ragas_scores = {}
    if ragas_data["question"]:
        try:
            dataset = Dataset.from_dict(ragas_data)
            ragas_result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision])
            ragas_scores = {
                "faithfulness": round(float(ragas_result["faithfulness"]), 4),
                "answer_relevancy": round(float(ragas_result["answer_relevancy"]), 4),
                "context_precision": round(float(ragas_result.get("context_precision", 0)), 4),
            }
        except Exception as e:
            ragas_scores = {"error": str(e)}

    n = len(test_queries)
    latencies.sort()
    results = {
        "n_queries": n,
        "task_completion_rate": round(completed / n, 4) if n else 0.0,
        "routing_accuracy": round(routing_correct / n, 4) if n else 0.0,
        "validation_pass_rate": round(sum(1 for r in run_records if r["validation_passed"]) / len(run_records), 4) if run_records else 0.0,
        "avg_cost_inr": round(sum(r["cost_inr"] for r in run_records) / len(run_records), 6) if run_records else 0.0,
        "latency_p50_s": round(latencies[int(len(latencies) * 0.50)], 3) if latencies else 0.0,
        "latency_p90_s": round(latencies[int(len(latencies) * 0.90)], 3) if latencies else 0.0,
        "ragas": ragas_scores,
        "per_query_results": run_records,
    }

    print(f"\nTask Completion: {results['task_completion_rate']:.1%} | "
          f"Routing Accuracy: {results['routing_accuracy']:.1%} | "
          f"p50 latency: {results['latency_p50_s']}s | "
          f"Avg cost: ₹{results['avg_cost_inr']:.4f}")
    if ragas_scores and "error" not in ragas_scores:
        print(f"Faithfulness: {ragas_scores['faithfulness']} | "
              f"Answer Relevancy: {ragas_scores['answer_relevancy']} | "
              f"Context Precision: {ragas_scores['context_precision']}")

    out_file = RESULTS_DIR / f"eval_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_file}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=20)
    args = parser.parse_args()
    run_evaluation(max_queries=args.max_queries)
