"""
Streamlit Metrics Dashboard
Live dashboard showing agent performance metrics updated after each run.

Run: streamlit run dashboard/app.py
"""
import json
import time
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
RESULTS_DIR = Path("data/eval/results")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #fafafa; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252836 100%);
        border: 1px solid #3d4255;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.3rem;
    }
    .metric-label { color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #f9fafb; font-size: 2rem; font-weight: 700; }
    .metric-delta-good { color: #34d399; font-size: 0.85rem; }
    .metric-delta-bad  { color: #f87171; font-size: 0.85rem; }
    .status-ok    { color: #34d399; }
    .status-error { color: #f87171; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Agent Dashboard")
    st.markdown("---")

    # API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.markdown('<span class="status-ok">● API Online</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-error">● API Offline</span>', unsafe_allow_html=True)
        health = {}

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")
    st.markdown("**Targets**")
    st.markdown("- Task Completion > 85%")
    st.markdown("- Faithfulness > 0.80")
    st.markdown("- Context Precision > 0.70")
    st.markdown("- p50 Latency < 4s")
    st.markdown("- Avg Cost < ₹0.84")


# ── Load metrics ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_live_metrics() -> List[dict]:
    """Load metrics from the live API."""
    try:
        r = requests.get(f"{API_URL}/metrics/json", timeout=3)
        return r.json().get("metrics", [])
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_eval_results() -> dict:
    """Load the latest evaluation results from disk."""
    if not RESULTS_DIR.exists():
        return {}
    result_files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
    if not result_files:
        return {}
    with open(result_files[0]) as f:
        return json.load(f)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🔬 Research Agent — Live Metrics")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")

live_metrics = load_live_metrics()
eval_results = load_eval_results()

# ── Top KPI row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

def kpi(col, label, value, target=None, good_direction="up", format_fn=str):
    with col:
        delta = ""
        if target is not None:
            if good_direction == "up":
                delta = "✅" if value >= target else "⚠️"
            else:
                delta = "✅" if value <= target else "⚠️"
        st.metric(label=label, value=format_fn(value), delta=delta)

# Use live metrics if available, fall back to eval results
if live_metrics:
    df_live = pd.DataFrame(live_metrics)
    n_total = len(df_live)
    n_validated = df_live["validation_passed"].sum() if "validation_passed" in df_live else 0
    avg_confidence = df_live["confidence"].mean() if "confidence" in df_live else 0
    avg_latency = df_live["latency_s"].mean() if "latency_s" in df_live else 0
    avg_cost = df_live["cost_inr"].mean() if "cost_inr" in df_live else 0
    avg_retries = df_live["retry_count"].mean() if "retry_count" in df_live else 0
else:
    n_total = eval_results.get("n_queries", 0)
    n_validated = int(n_total * eval_results.get("validation_pass_rate", 0))
    avg_confidence = 0.0
    avg_latency = eval_results.get("latency_p50_s", 0)
    avg_cost = eval_results.get("avg_cost_inr", 0)
    avg_retries = eval_results.get("avg_retries_per_query", 0)

completion_rate = eval_results.get("task_completion_rate", n_validated / max(n_total, 1))

kpi(col1, "Queries Run", n_total, format_fn=str)
kpi(col2, "Completion Rate", f"{completion_rate:.1%}", target="85%")
kpi(col3, "Avg Confidence", f"{avg_confidence:.2f}", target=0.7, format_fn=lambda x: f"{x:.2f}")
kpi(col4, "p50 Latency", f"{avg_latency:.1f}s")
kpi(col5, "Avg Cost", f"₹{avg_cost:.4f}")
kpi(col6, "Avg Retries", f"{avg_retries:.2f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Metrics", "🧪 RAGAS Evaluation", "⚡ Latency", "💬 Query Explorer"])

# ── Tab 1: Live Metrics ───────────────────────────────────────────────────────
with tab1:
    if not live_metrics:
        st.info("No live metrics yet. Run some queries via the API to see data here.")
    else:
        df = pd.DataFrame(live_metrics)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Query Type Distribution")
            if "query_type" in df.columns:
                type_counts = df["query_type"].value_counts().reset_index()
                type_counts.columns = ["query_type", "count"]
                fig = px.pie(
                    type_counts, values="count", names="query_type",
                    color_discrete_sequence=["#6366f1", "#22d3ee", "#f59e0b", "#34d399"],
                    hole=0.4,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Confidence Distribution")
            if "confidence" in df.columns:
                fig = px.histogram(
                    df, x="confidence", nbins=20,
                    color_discrete_sequence=["#6366f1"],
                )
                fig.add_vline(x=0.7, line_dash="dash", line_color="#f59e0b", annotation_text="Pass threshold")
                fig.add_vline(x=0.5, line_dash="dash", line_color="#f87171", annotation_text="Clarify threshold")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cost & Latency Over Time")
        c3, c4 = st.columns(2)
        with c3:
            if "cost_inr" in df.columns:
                fig = px.line(df, x="timestamp", y="cost_inr", color_discrete_sequence=["#22d3ee"])
                fig.add_hline(y=0.84, line_dash="dash", line_color="#f59e0b", annotation_text="₹0.84 target")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            if "latency_s" in df.columns:
                fig = px.line(df, x="timestamp", y="latency_s", color_discrete_sequence=["#34d399"])
                fig.add_hline(y=4.0, line_dash="dash", line_color="#f59e0b", annotation_text="4s target")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Validation Pass Rate Over Time")
        if "validation_passed" in df.columns:
            df["pass_rolling"] = df["validation_passed"].astype(float).rolling(10, min_periods=1).mean()
            fig = px.line(df, x="timestamp", y="pass_rolling", color_discrete_sequence=["#f59e0b"])
            fig.add_hline(y=0.85, line_dash="dash", line_color="#34d399", annotation_text="85% target")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa",
                              yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: RAGAS Evaluation ───────────────────────────────────────────────────
with tab2:
    if not eval_results:
        st.info("No evaluation results found. Run `python scripts/evaluate.py` first.")
    else:
        ragas = eval_results.get("ragas", {})
        if ragas and "error" not in ragas:
            c1, c2, c3 = st.columns(3)
            metrics = [
                ("Faithfulness", ragas.get("faithfulness", 0), 0.80),
                ("Answer Relevancy", ragas.get("answer_relevancy", 0), 0.75),
                ("Context Precision", ragas.get("context_precision", 0), 0.70),
            ]
            for col, (name, val, target) in zip([c1, c2, c3], metrics):
                with col:
                    color = "#34d399" if val >= target else "#f87171"
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value" style="color:{color}">{val:.3f}</div>
                        <div>Target: {target}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Task Completion", f"{eval_results.get('task_completion_rate', 0):.1%}")
            c2.metric("Routing Accuracy", f"{eval_results.get('routing_accuracy', 0):.1%}")
            c3.metric("Validation Pass Rate", f"{eval_results.get('validation_pass_rate', 0):.1%}")
            c4.metric("Avg Cost/Query", f"₹{eval_results.get('avg_cost_inr', 0):.4f}")

        per_query = eval_results.get("per_query_results", [])
        if per_query:
            st.subheader("Per-Query Results")
            df_eval = pd.DataFrame(per_query)
            st.dataframe(df_eval, use_container_width=True)

# ── Tab 3: Latency Breakdown ──────────────────────────────────────────────────
with tab3:
    if not live_metrics:
        st.info("Run queries to see latency breakdown by node.")
    else:
        df = pd.DataFrame(live_metrics)
        if "node_latencies" in df.columns:
            node_data = []
            for _, row in df.iterrows():
                for node, lat in (row["node_latencies"] or {}).items():
                    node_data.append({"node": node, "latency_s": lat})

            if node_data:
                df_nodes = pd.DataFrame(node_data)
                agg = df_nodes.groupby("node")["latency_s"].agg(["mean", "median", "max"]).reset_index()
                agg.columns = ["node", "mean", "p50", "max"]

                fig = px.bar(
                    agg.melt(id_vars="node", value_vars=["mean", "p50", "max"],
                             var_name="stat", value_name="latency_s"),
                    x="node", y="latency_s", color="stat", barmode="group",
                    color_discrete_sequence=["#6366f1", "#22d3ee", "#f87171"],
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fafafa")
                st.plotly_chart(fig, use_container_width=True)

        if "latency_s" in df.columns:
            st.subheader("Latency Percentiles")
            sorted_lat = df["latency_s"].sort_values()
            c1, c2, c3 = st.columns(3)
            c1.metric("p50 Latency", f"{sorted_lat.quantile(0.50):.2f}s")
            c2.metric("p90 Latency", f"{sorted_lat.quantile(0.90):.2f}s")
            c3.metric("p99 Latency", f"{sorted_lat.quantile(0.99):.2f}s")

# ── Tab 4: Query Explorer ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Try the Agent")
    query_input = st.text_area("Enter a query:", placeholder="How does LoRA reduce memory compared to full fine-tuning?", height=100)
    session_id = st.text_input("Session ID (optional):", value="dashboard-session")

    col_q, col_s = st.columns([3, 1])
    with col_q:
        run_btn = st.button("▶ Run Query", type="primary")
    with col_s:
        stream_btn = st.button("⚡ Stream Query")

    if run_btn and query_input:
        with st.spinner("Agent thinking..."):
            start = time.time()
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": query_input, "session_id": session_id},
                    timeout=60,
                )
                result = resp.json()
                elapsed = time.time() - start

                st.success(f"✅ Done in {elapsed:.2f}s")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Query Type", result.get("query_type", ""))
                c2.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                c3.metric("Validated", "✅" if result.get("validation_passed") else "❌")
                c4.metric("Cost", f"₹{result.get('cost_inr', 0):.4f}")

                if result.get("awaiting_clarification"):
                    st.warning(f"🤔 Clarification needed: {result.get('final_answer', '')}")
                else:
                    st.markdown("### Answer")
                    st.markdown(result.get("final_answer", ""))

                    if result.get("cited_sources"):
                        with st.expander("📚 Sources"):
                            for src in result["cited_sources"]:
                                st.markdown(f"- {src}")

                    with st.expander("🔍 Node Latencies"):
                        st.json(result.get("node_latencies", {}))

            except Exception as e:
                st.error(f"API error: {e}")
