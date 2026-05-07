"""
Streamlit Web Interface for Urological Oncology RAG System
Professional UI - API-connected v4
"""

import streamlit as st
import os
import time
import json
import uuid
import requests
from pathlib import Path
import plotly.graph_objects as go

# ── Backend API configuration ──────────────────────────────────────────────────
_API_BASE = os.environ.get("API_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.environ.get("API_KEY", "")

_QUALITY_BADGES: dict[str, tuple[str, str]] = {
    "high":         ("#2e7d32", "🟢 High Evidence"),
    "hedged":       ("#f57c00", "🟡 Hedged"),
    "caveated":     ("#c62828", "🔴 Caveated"),
    "insufficient": ("#546e7a", "⚫ Insufficient"),
}

_DESIGN_COLORS: dict[str, str] = {
    "rct":           "#2ca02c",
    "meta_analysis": "#1f77b4",
    "cohort":        "#ff7f0e",
    "case_report":   "#9467bd",
    "review":        "#8c564b",
    "unknown":       "#7f7f7f",
}

st.set_page_config(
    page_title="Urological Oncology RAG",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1.5rem;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
    }
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
        margin-top: 0.5rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .citation {
        color: #1f77b4;
        font-weight: 600;
        text-decoration: none;
        cursor: pointer;
        padding: 2px 6px;
        background: #e8f4f8;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .citation:hover {
        background: #d0e8f0;
    }
    .stButton button {
        width: 100%;
    }
    .stTextArea textarea {
        min-height: 80px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'quality_history' not in st.session_state:
    st.session_state.quality_history = []
if 'session_metrics' not in st.session_state:
    st.session_state.session_metrics = {'queries': [], 'latencies': [], 'cache_hits': 0}
if 'top_k' not in st.session_state:
    st.session_state.top_k = 5
if 'show_context' not in st.session_state:
    st.session_state.show_context = False
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = None


# ── Backend helpers ────────────────────────────────────────────────────────────

def _query_backend(
    query: str,
    cancer_types: list,
    top_k: int,
    conversation_id: str | None,
) -> dict:
    api_key = st.session_state.get("user_api_key") or _API_KEY
    headers = {"X-API-Key": api_key} if api_key else {}
    payload: dict = {"query": query, "cancer_types": cancer_types, "top_k": top_k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = requests.post(f"{_API_BASE}/query", json=payload, headers=headers, timeout=180)
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {detail}",
            response=resp,
        )
    return resp.json()


def _quality_badge_html(evidence_quality: str) -> str:
    color, label = _QUALITY_BADGES.get(
        evidence_quality, ("#546e7a", evidence_quality.replace("_", " ").title())
    )
    return (
        f"<span style='background:{color};color:white;padding:3px 12px;"
        f"border-radius:12px;font-size:0.85em;font-weight:600'>{label}</span>"
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def display_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🔑 API Access")
        user_api_key = st.text_input(
            "Access Key",
            type="password",
            placeholder="Provided by administrator",
            help="Enter the access key provided to you.",
            key="api_key_input"
        )
        if user_api_key:
            st.session_state.user_api_key = user_api_key
            st.success("✅ Key active — Unlimited queries")
        else:
            st.session_state.user_api_key = None

        st.divider()
        st.markdown("### 📊 Session Stats")
        metrics = st.session_state.session_metrics
        queries_count = len(metrics['queries'])
        if queries_count > 0:
            latencies = metrics['latencies']
            avg_latency = sum(latencies) / len(latencies)
            cache_hits = metrics['cache_hits']
            cache_rate = cache_hits / queries_count * 100
            st.markdown(f"""
```
Queries:     {queries_count}
Avg Latency: {avg_latency:.2f}s
Cache Hits:  {cache_hits}/{queries_count} ({cache_rate:.0f}%)
Fastest:     {min(latencies):.2f}s
Slowest:     {max(latencies):.2f}s
```
            """)
        else:
            st.info("No queries yet in this session.")


# ── Query helpers ──────────────────────────────────────────────────────────────

def format_answer_with_citations(answer: str, sources: list) -> str:
    formatted = answer
    for i, source in enumerate(sources):
        tag = f"[Doc {i + 1}]"
        title = (source.get("title") or "")[:60]
        if tag in formatted:
            formatted = formatted.replace(
                tag,
                f'<span class="citation" title="{title}...">{tag}</span>',
            )
    return formatted


def display_sources(sources: list, show_context: bool) -> None:
    st.markdown("### 📚 Sources")
    for idx, source in enumerate(sources, 1):
        title = (source.get("title") or "Unknown")[:70]
        year = source.get("year", "")
        design = source.get("study_design", "unknown")
        sample_size = source.get("sample_size")
        pmid = source.get("pmid", "")
        authors = source.get("authors", "")
        journal = source.get("journal", "")
        key_finding = source.get("key_finding", "")
        header = f"**[{idx}] {title}**" + (f"  ({year})" if year else "")
        with st.expander(header, expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption("**Study Design**")
                color = _DESIGN_COLORS.get(design, "#7f7f7f")
                st.markdown(
                    f"<span style='background:{color};color:white;padding:2px 8px;"
                    f"border-radius:10px;font-size:0.8em'>"
                    f"{design.replace('_', ' ').title()}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.caption("**Sample Size**")
                st.text(str(sample_size) if sample_size else "N/A")
            with col3:
                st.caption("**Section**")
                st.text(source.get("section") or "N/A")
            with col4:
                st.caption("**PMID**")
                if pmid:
                    st.markdown(f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})")
                else:
                    st.text("N/A")
            if authors or journal:
                st.caption(f"_{authors}. {journal} {year}_".strip(". "))
            st.divider()
            if show_context:
                st.caption("**Key Finding:**")
                st.text(key_finding)
            else:
                st.caption("**Preview:**")
                st.text(key_finding[:200] + ("…" if len(key_finding) > 200 else ""))


def evaluate_response_quality(query: str, answer: str, sources: list) -> dict:
    from src.evaluation.judges import JudgeSet

    class _Chunk:
        def __init__(self, text: str, metadata: dict):
            self.text = text
            self.metadata = metadata

    chunks = [
        _Chunk(
            text=s.get("key_finding", ""),
            metadata={"evidence_level": 2, "study_design": s.get("study_design", "unknown")},
        )
        for s in sources
    ]
    scores = JudgeSet().score_all(question=query, answer=answer, chunks=chunks)
    return {
        "faithfulness": scores.faithfulness,
        "relevance":    scores.answer_relevance,
        "precision":    scores.context_precision,
    }


def create_metrics_gauge(value: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "lightgreen"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ── System Performance tab ─────────────────────────────────────────────────────

def display_system_performance_tab() -> None:
    st.markdown("## 📊 System Performance")

    metrics = st.session_state.quality_metrics
    history = st.session_state.quality_history

    if not metrics:
        st.info("Run a query to see real-time quality metrics for each response.")
    else:
        overall = (metrics['faithfulness'] + metrics['relevance'] + metrics['precision']) / 3
        if overall >= 0.9:
            st.success(f"✅ Excellent quality: {overall:.1%}")
        elif overall >= 0.8:
            st.info(f"✅ Good quality: {overall:.1%}")
        else:
            st.warning(f"⚠️ Quality score: {overall:.1%}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(
                create_metrics_gauge(metrics['faithfulness'], "Faithfulness"),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.caption("Is the answer grounded in sources?")
        with col2:
            st.plotly_chart(
                create_metrics_gauge(metrics['relevance'], "Relevance"),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.caption("Does the answer address the question?")
        with col3:
            st.plotly_chart(
                create_metrics_gauge(metrics['precision'], "Context Precision"),
                use_container_width=True,
                config={'displayModeBar': False},
            )
            st.caption("Are the retrieved sources relevant?")

    # ── Session history ────────────────────────────────────────────────────────
    if len(history) > 1:
        st.divider()
        st.markdown("### 📈 Quality Over This Session")
        overall_scores = [
            (h['faithfulness'] + h['relevance'] + h['precision']) / 3
            for h in history
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=[s * 100 for s in overall_scores],
            mode='lines+markers',
            name='Overall',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=[h['faithfulness'] * 100 for h in history],
            mode='lines+markers',
            name='Faithfulness',
            line=dict(color='#2ca02c', width=1.5, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=[h['relevance'] * 100 for h in history],
            mode='lines+markers',
            name='Relevance',
            line=dict(color='#ff7f0e', width=1.5, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=[h['precision'] * 100 for h in history],
            mode='lines+markers',
            name='Context Precision',
            line=dict(color='#9467bd', width=1.5, dash='dot'),
        ))
        fig.update_layout(
            xaxis_title="Query #",
            yaxis_title="Score (%)",
            yaxis_range=[0, 100],
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── How quality is measured ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔬 How Quality Is Measured")
    st.caption(
        "Each response is automatically evaluated using heuristic judges. "
        "No external API call is made — evaluation runs locally in under a second."
    )

    with st.expander("**Faithfulness** — Is the answer grounded in sources?", expanded=True):
        st.write("""
        Faithfulness measures whether the generated answer is traceable to the retrieved chunks,
        not fabricated. The judge penalises:

        - **Unsupported clinical directives** — phrases like *"you should take X mg"* without
          a `[Doc N]` citation score lower
        - **Missing citations** — answers with no inline references to retrieved sources
          score lower
        - **Specificity without grounding** — numerical claims (doses, survival rates) not
          appearing in any chunk

        A score of **1.0** means every factual claim in the answer is backed by a source.
        """)

    with st.expander("**Relevance** — Does the answer address the question?", expanded=True):
        st.write("""
        Relevance measures how directly the answer responds to what was asked.
        The judge checks:

        - **Keyword overlap** between the question's key terms and the answer
        - **Length appropriateness** — very short or evasive answers score lower;
          padding-heavy answers without substance also score lower
        - **Direct address** — answers that restate the question without answering it
          are penalised

        A score of **1.0** means the answer is precisely responsive to the query.
        """)

    with st.expander("**Context Precision** — Are the retrieved sources on-topic?", expanded=True):
        st.write("""
        Context Precision evaluates the quality of the retrieved chunks before generation.
        It measures:

        - **Keyword overlap** between each source chunk and the original query
        - **Evidence diversity** — results drawing from multiple independent papers
          score higher than repeated citations of a single source
        - **Section relevance** — chunks from results, conclusions, and abstract sections
          score higher than background or methods sections

        A score of **1.0** means every retrieved chunk is highly relevant to the query.
        """)


# ── About tab ──────────────────────────────────────────────────────────────────

def display_about_tab() -> None:
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.subheader("✨ Key Features")

        with st.expander("📚 Comprehensive Knowledge Base", expanded=False):
            st.write("""
            - **27,500+ full-text papers** from PubMed Central Open Access
            - **6 cancer types:** Prostate, Bladder, Kidney, Testicular, Penile, Adrenal
            - **685,000+ section-aware chunks** for precise retrieval
            - **Years covered:** 2010–2025 (latest high-evidence research)
            - **Filtered corpus:** RCTs, meta-analyses, systematic reviews, clinical guidelines
            """)

        with st.expander("🎯 Advanced RAG Pipeline", expanded=False):
            st.write("""
            **Retrieval:** Hybrid semantic + keyword search (OpenAI text-embedding-3-small + BM25)

            **Reranking:** Cohere cross-encoder reranker for precision before generation

            **Generation:** Anthropic Claude produces answers grounded in retrieved sources

            **Speed:** BM25 disk cache delivers near-instant startup; query cache for repeats
            """)

        with st.expander("💬 Conversation Memory", expanded=False):
            st.write("""
            - Multi-turn conversations with context awareness
            - Automatic query rewriting using conversation history
            - Maintains relevance across follow-up questions
            - Enable via the Chat Mode toggle in the Query tab
            """)

        with st.expander("📊 Quality Evaluation", expanded=False):
            st.write("""
            - **Faithfulness:** Measures answer grounding in retrieved sources
            - **Relevance:** Ensures the answer directly addresses the question
            - **Context Precision:** Validates that retrieved chunks are on-topic
            - Runs automatically after every query — see the System Performance tab
            """)

        with st.expander("🔗 Citation Tracking", expanded=False):
            st.write("""
            - Inline `[Doc N]` citations linked to source papers
            - Direct links to PubMed entries for every source
            - Section-level attribution (Results, Methods, Abstract…)
            - Expandable source cards with study design, sample size, and key finding
            """)

        st.markdown("---")
        st.subheader("🔑 Access")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔐 API Key**
            - Unlimited queries
            - Contact the administrator to obtain a key
            - Enter it in the sidebar to activate
            """)
        with col2:
            st.markdown("""
            **🔒 Security**
            - Keys managed per user
            - Rate limiting enforced server-side
            - All queries logged for quality monitoring
            """)

    with right_col:
        st.subheader("📊 Dataset at a Glance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", "27,500+", help="Full-text peer-reviewed articles")
            st.metric("Chunks", "685,000+", help="Section-aware text segments")
            st.metric("Avg Latency", "~7s", help="End-to-end query response time")
        with col2:
            st.metric("Topics", "6", help="Urological cancer types covered")
            st.metric("Evidence Filter", "RCT+", help="High-evidence studies only")
            st.metric("Years", "2010–2025", help="Publication date range")

        st.markdown("---")
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        <div style='display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.75rem 0;'>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Python 3.10</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Anthropic Claude</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>OpenAI Embeddings</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Qdrant</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>FastAPI</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Cohere Rerank</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>BM25</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Streamlit</span>
            <span style='background:#1f77b4;color:white;padding:0.25rem 0.7rem;border-radius:15px;font-size:0.82rem;'>Docker</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔗 Links")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("📂 GitHub", "https://github.com/steph-grigors/urological-oncology-rag-prod", use_container_width=True)
            st.link_button("💼 Portfolio", "https://www.stephan-gs.work", use_container_width=True)
        with col2:
            st.link_button("🔗 LinkedIn", "https://linkedin.com/in/stéphan-grs", use_container_width=True)
            st.link_button("📧 Contact", "mailto:stephan.grigorescu@gmail.com", use_container_width=True)

    st.divider()
    st.subheader("📖 Data Sources")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    with c1:
        st.markdown("**Prostate Cancer**\n- 15,366 papers\n- 387,650 chunks")
    with c2:
        st.markdown("**Bladder Cancer**\n- 4,779 papers\n- 119,908 chunks")
    with c3:
        st.markdown("**Kidney Cancer**\n- 5,244 papers\n- 129,176 chunks")
    with c4:
        st.markdown("**Testicular Cancer**\n- 686 papers\n- 16,140 chunks")
    with c5:
        st.markdown("**Adrenal Cancer**\n- 1,185 papers\n- 26,801 chunks")
    with c6:
        st.markdown("**Penile Cancer**\n- 255 papers\n- 5,789 chunks")
    st.caption("All papers sourced from PubMed Central Open Access Subset")

    st.divider()
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0;'>
        <h3>👨‍💻 Developed by Stéphan Grigorescu</h3>
        <p style='color:#666;'>Data Scientist & AI Engineer | NLP · RAG Systems · Medical AI</p>
    </div>
    """, unsafe_allow_html=True)

    disc_col1, disc_col2 = st.columns(2)
    with disc_col1:
        with st.expander("⚖️ Legal Disclaimer"):
            st.caption("""
            This application is for educational and research purposes only. It is NOT a
            substitute for professional medical advice, diagnosis, or treatment. Always
            consult qualified health providers. AI-generated answers should be verified
            against original sources before clinical application.
            """)
    with disc_col2:
        with st.expander("📜 Data Attribution"):
            st.caption("""
            All research papers are sourced from the PubMed Central Open Access Subset
            under Creative Commons licenses. Citations and links to original papers are
            provided for all responses, in accordance with NLM attribution requirements.
            """)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    display_sidebar()

    st.markdown("""
    <div style='margin-top:0.5rem;margin-bottom:1.5rem;'>
        <div class='app-title'>🔬 Urological Oncology RAG System</div>
        <div class='app-subtitle'>Evidence-based medical research powered by AI retrieval-augmented generation</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 System Performance", "ℹ️ About"])

    # ── Tab 1: Query ───────────────────────────────────────────────────────────
    with tab1:

        # What is This / How It Works
        info_col1, info_col2 = st.columns([1, 1], gap="large")
        with info_col1:
            st.subheader("💡 What is This?")
            st.write("""
            An AI-powered research assistant providing evidence-based answers from **27,500+**
            peer-reviewed papers across **6 urological cancer types**. Uses advanced RAG
            architecture to deliver accurate, cited responses with zero hallucination.
            """)
        with info_col2:
            st.subheader("🔄 How It Works")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.markdown(
                    "<div style='text-align:center'>"
                    "<div style='font-size:1.8rem'>❓</div>"
                    "<div style='font-weight:bold'>Ask</div>"
                    "<div style='font-size:0.8rem;color:#666'>Your question</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with fc2:
                st.markdown(
                    "<div style='text-align:center'>"
                    "<div style='font-size:1.8rem'>🔍</div>"
                    "<div style='font-weight:bold'>Search</div>"
                    "<div style='font-size:0.8rem;color:#666'>685K+ chunks</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with fc3:
                st.markdown(
                    "<div style='text-align:center'>"
                    "<div style='font-size:1.8rem'>🤖</div>"
                    "<div style='font-weight:bold'>Generate</div>"
                    "<div style='font-size:0.8rem;color:#666'>AI answer</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with fc4:
                st.markdown(
                    "<div style='text-align:center'>"
                    "<div style='font-size:1.8rem'>📚</div>"
                    "<div style='font-weight:bold'>Cite</div>"
                    "<div style='font-size:0.8rem;color:#666'>Source papers</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # Query section
        st.subheader("I am your AI urologist — Ask me a question!")

        topic_filter = st.selectbox(
            "Search in:",
            [
                "All Topics",
                "Prostate Cancer",
                "Bladder Cancer",
                "Kidney Cancer",
                "Testicular Cancer",
                "Penile Cancer",
                "Adrenal Cancer",
            ],
            index=0,
            help="Filter results to a specific cancer type",
        )

        query = st.text_area(
            "Your question:",
            value=st.session_state.get("query", ""),
            height=200,
            placeholder="Ask about treatment options, diagnosis, biomarkers, side effects, clinical trials…",
            key="query_input",
            label_visibility="collapsed",
        )

        # Search + Clear
        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col1:
            search_button = st.button("🚀 Search", type="primary", use_container_width=True)
        with btn_col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        # Chat controls
        chat_col1, chat_col2 = st.columns([1, 1])
        with chat_col1:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.current_response = None
                st.session_state.quality_metrics = None
                st.rerun()
        with chat_col2:
            context_mode = st.checkbox(
                "💬 Enable Chat Mode",
                value=st.session_state.conversation_id is not None,
                help="Multi-turn conversation with context awareness",
            )

        if context_mode and st.session_state.conversation_id is None:
            st.session_state.conversation_id = str(uuid.uuid4())
        elif not context_mode:
            st.session_state.conversation_id = None

        if st.session_state.conversation_id:
            st.caption("💬 Chat mode active")

        # Query settings
        with st.expander("⚙️ Query Settings", expanded=False):
            st.session_state.top_k = st.slider(
                "Number of sources",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Number of relevant chunks retrieved before reranking",
            )
            st.session_state.show_context = st.checkbox(
                "Show full source text",
                value=st.session_state.show_context,
                help="Show complete key finding instead of a short preview",
            )

        # Execute
        if clear_button:
            st.session_state["query"] = ""
            st.session_state.current_response = None
            st.session_state.quality_metrics = None
            st.rerun()

        if search_button and query:
            query = query.strip()
            with st.spinner("🔍 Searching knowledge base…"):
                start_time = time.time()
                try:
                    cancer_filter = (
                        []
                        if topic_filter == "All Topics"
                        else [topic_filter.lower().replace(" cancer", "").strip()]
                    )
                    response = _query_backend(
                        query=query,
                        cancer_types=cancer_filter,
                        top_k=st.session_state.top_k,
                        conversation_id=st.session_state.conversation_id,
                    )
                    latency_ms = response.get("latency_ms", {})
                    latency = latency_ms.get("total", (time.time() - start_time) * 1000) / 1000

                    st.session_state.current_response = {
                        "query":            query,
                        "answer":           response.get("answer", ""),
                        "sources":          response.get("sources", []),
                        "num_sources":      len(response.get("sources", [])),
                        "latency":          latency,
                        "evidence_quality": response.get("evidence_quality", "insufficient"),
                        "confidence_score": response.get("confidence_score", 0.0),
                    }

                    # Auto quality evaluation
                    try:
                        metrics = evaluate_response_quality(
                            query,
                            response.get("answer", ""),
                            response.get("sources", []),
                        )
                        st.session_state.quality_metrics = metrics
                        st.session_state.quality_history.append({**metrics, "query": query})
                    except Exception:
                        st.session_state.quality_metrics = None

                    st.session_state.session_metrics["queries"].append(query)
                    st.session_state.session_metrics["latencies"].append(latency)
                    if latency < 0.5:
                        st.session_state.session_metrics["cache_hits"] += 1

                except Exception as e:
                    import traceback
                    err_msg = str(e)
                    short_msg = err_msg.split(":")[0] if ":" in err_msg else err_msg
                    st.error(f"❌ Query failed — {short_msg}. See debug info below.")
                    with st.expander("🐛 Debug Info", expanded=False):
                        st.code(err_msg)
                        st.code(traceback.format_exc())

        elif search_button:
            st.warning("⚠️ Please enter a question")

        # Results
        if st.session_state.current_response:
            resp = st.session_state.current_response
            st.divider()
            badge_html = _quality_badge_html(resp.get("evidence_quality", "insufficient"))
            st.markdown(
                f"✅ Found **{resp['num_sources']}** sources in **{resp['latency']:.2f}s**"
                f" &nbsp; {badge_html}",
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("### 📄 Answer")
            st.markdown(
                format_answer_with_citations(resp["answer"], resp["sources"]),
                unsafe_allow_html=True,
            )
            st.divider()
            display_sources(resp["sources"], st.session_state.show_context)

    # ── Tab 2: System Performance ──────────────────────────────────────────────
    with tab2:
        display_system_performance_tab()

    # ── Tab 3: About ───────────────────────────────────────────────────────────
    with tab3:
        display_about_tab()


if __name__ == "__main__":
    main()
