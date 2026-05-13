"""
Multi-turn intent slot-filling test.
Simulates the full conversation loop:
  Turn 1: Ambiguous → clarification asked
  Turn 2: Short answer → slot filled, still ambiguous
  Turn 3: Short answer → all slots filled, resolves to complex query, retrieval fires
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dotenv import load_dotenv; load_dotenv()
from agent.graph import run_query

SESSION = "test-slot-fill-01"

print("=" * 60)
print("SCENARIO A: Simple (Wi-Fi Password)")
print("=" * 60)
r = run_query("What is the guest Wi-Fi password?", session_id="test-simple-01")
print(f"  Type:   {r.get('query_type')}")
print(f"  Answer: {r.get('final_answer', '')[:120]}")

print()
print("=" * 60)
print("SCENARIO B: Multi-Turn Slot Filling (The Amnesia Loop Test)")
print("=" * 60)

print("\n[Turn 1] Ambiguous intent")
r1 = run_query("Which clients should we focus on?", session_id=SESSION)
print(f"  Type:     {r1.get('query_type')}")
print(f"  Question: {r1.get('final_answer', '') or r1.get('clarification_question','')}")
print(f"  Slots:    {r1.get('intent_slots')}")

print("\n[Turn 2] User picks metric — short answer")
r2 = run_query("fastest growth potential", session_id=SESSION)
print(f"  Type:     {r2.get('query_type')}")
if r2.get('query_type') == 'awaiting_clarification':
    print(f"  Question: {r2.get('final_answer', '')}")
else:
    print(f"  Answer:   {r2.get('final_answer','')[:120]}")
print(f"  Slots:    {r2.get('intent_slots')}")

print("\n[Turn 3] User fills entity — should RESOLVE and retrieve")
r3 = run_query("clients", session_id=SESSION)
print(f"  Type:     {r3.get('query_type')}")
print(f"  Slots:    {r3.get('intent_slots')}")
print(f"  Resolved: {r3.get('original_query','')[:120]}")
print(f"  Answer:   {r3.get('final_answer','')[:200]}")

print()
print("=" * 60)
print("SCENARIO C: Complex Query (Churn Risk)")
print("=" * 60)
r_c = run_query(
    "Which clients are at high churn risk and what are the specific reasons?",
    session_id="test-complex-01"
)
print(f"  Type:       {r_c.get('query_type')}")
print(f"  Confidence: {r_c.get('confidence')}")
print(f"  Validation: {r_c.get('validation_passed')}")
print(f"  Retries:    {r_c.get('retry_count')}")
print(f"  Answer:     {r_c.get('final_answer','')[:200]}...")
