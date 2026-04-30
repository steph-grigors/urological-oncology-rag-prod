"""
Streamlit Web Interface for Urological Oncology RAG System
Professional UI - Refactored v2
"""

import streamlit as st
import os
import time
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from src.retrieval_optimized import OptimizedRAGRetriever, OptimizedRetrievalConfig
from src.conversation_memory import ConversationMemory, ConversationalRAG

# Page configuration
st.set_page_config(
    page_title="Urological Oncology RAG",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem;
    }

    /* Reduce sidebar top padding */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
    }

    /* Fix sidebar width */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 360px !important;
        max-width: 360px !important;
    }

    /* Adjust main content margin when sidebar is fixed width */
    .main .block-container {
        max-width: calc(100% - 360px);
    }

    /* Compact headers */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    /* Citation styling */
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

    /* Source cards */
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin-bottom: 0.5rem;
    }

    /* Metrics */
    .metric-box {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }

    /* Buttons */
    .stButton button {
        width: 100%;
    }

    /* Compact dividers */
    .compact-divider {
        margin: 0.5rem 0;
    }

    /* Reduce vertical gaps */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }

    /* Compact text areas */
    .stTextArea textarea {
        min-height: 80px !important;
    }

    /* Remove excessive padding from tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-top: 0.5rem;
    }

    /* Reduce spacing in sidebar elements */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'session_metrics' not in st.session_state:
    st.session_state.session_metrics = {
        'queries': [],
        'latencies': [],
        'cache_hits': 0
    }
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = None


@st.cache_resource
def load_rag_system(_api_key=None):
    """Load RAG system with optional user API key"""

    # Use user key if provided, otherwise use environment variable
    api_key = _api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ No API key available. Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Set the API key for this session
    os.environ["OPENAI_API_KEY"] = api_key

    with st.spinner("🚀 Initializing RAG System..."):
        retriever = OptimizedRAGRetriever(
            chroma_db_dir="chroma_db_scaled",
            collection_name="urological_oncology_papers",
            config=OptimizedRetrievalConfig(
                top_k=5,
                max_context_length=3000,
                max_tokens=500,
                use_cache=True
            )
        )
    return retriever


@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics if available"""
    try:
        metrics_path = Path("data/evaluation/scaled_system_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def display_sidebar():
    """Minimalist sidebar with live session stats"""
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")

        # API Key input
        user_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your key for unlimited usage. Leave empty for 2 free queries.",
            key="api_key_input"
        )

        # Store in session state and reset counter if key added
        if user_api_key:
            st.session_state.user_api_key = user_api_key
            st.session_state.query_count = 0  # Reset counter
            st.success("✅ Your key active - Unlimited queries")
        else:
            st.session_state.user_api_key = None
            free_remaining = 2 - st.session_state.query_count  # Changed from 1 to 2
            if free_remaining > 0:
                # Use correct plural/singular
                query_text = "queries" if free_remaining > 1 else "query"
                st.info(f"ℹ️ Demo mode: {free_remaining} free {query_text} remaining")
            else:
                st.error("❌ Free queries used - Add API key above")

        st.divider()

        st.markdown("### 🔬 System Information")

        # Live session metrics
        st.markdown("#### Current Session")

        metrics = st.session_state.session_metrics
        queries_count = len(metrics['queries'])

        if queries_count > 0:
            latencies = metrics['latencies']
            avg_latency = sum(latencies) / len(latencies)
            cache_hits = metrics['cache_hits']
            cache_rate = (cache_hits / queries_count * 100) if queries_count > 0 else 0

            st.markdown(f"""
```
            Queries:      {queries_count}
            Avg Latency:  {avg_latency:.2f}s
            Cache Hits:   {cache_hits}/{queries_count} ({cache_rate:.0f}%)
            Fastest:      {min(latencies):.2f}s
            Slowest:      {max(latencies):.2f}s
```
            """)
        else:
            st.info("No queries yet")

        st.divider()

        # Query Settings
        st.markdown("### ⚙️ Query Settings")

        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )

        model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model for answer generation"
        )

        show_context = st.checkbox(
            "Show full context",
            value=False,
            help="Show complete source text instead of short preview"
        )

        return top_k, model, show_context

def format_answer_with_citations(answer, sources):
    """Format answer with inline citation links and hover tooltips"""
    formatted_answer = answer

    # Add citation styling
    for i in range(len(sources)):
        citation = f"[{i+1}]"
        if citation in formatted_answer:
            formatted_answer = formatted_answer.replace(
                citation,
                f'<span class="citation" title="{sources[i]["title"][:60]}...">{citation}</span>'
            )

    return formatted_answer


def display_sources(sources, show_context):
    """Display sources in collapsible expandable format"""
    st.markdown("### 📚 Sources")

    for idx, source in enumerate(sources, 1):
        with st.expander(
            f"**[{idx}] {source['title'][:70]}...**  "
            f"(Relevance: {source['similarity']:.0%})",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("**Section**")
                st.text(source['section'])

            with col2:
                st.caption("**PMID**")
                if source['pmid']:
                    st.markdown(f"[{source['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{source['pmid']})")
                else:
                    st.text("N/A")

            with col3:
                st.caption("**DOI**")
                st.text(source['doi'] if source['doi'] else "N/A")

            st.divider()

            if show_context:
                st.caption("**Full Context:**")
                st.text(source['text_preview'])
            else:
                st.caption("**Preview:**")
                st.text(source['text_preview'][:200] + "...")


def evaluate_response_quality(retriever, query, answer, sources):
    """Evaluate response quality on-demand"""
    from src.evaluation import RAGEvaluator

    evaluator = RAGEvaluator(rag_retriever=retriever)

    # Prepare context
    context = "\n\n".join([
        f"[Doc {i+1}]\n{source['text_preview'][:500]}"
        for i, source in enumerate(sources)
    ])

    # Evaluate
    faithfulness = evaluator.evaluate_faithfulness(query, answer, context)
    relevance = evaluator.evaluate_relevance(query, answer)

    # Prepare chunks for precision
    chunks = [{'text': s['text_preview']} for s in sources]
    precision = evaluator.evaluate_context_precision(query, chunks)

    return {
        'faithfulness': faithfulness,
        'relevance': relevance,
        'precision': precision
    }


def create_metrics_gauge(value, title):
    """Create a gauge chart for a metric"""
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
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def display_metrics_dashboard():
    """Display system-wide evaluation metrics from batch testing"""
    st.markdown("## 📊 System Performance")
    st.caption("Based on 12 test queries across 4 cancer types")

    metrics = load_evaluation_metrics()

    if not metrics:
        st.warning("⚠️ No evaluation metrics found. Run: `python -m src.evaluate_scaled_system`")
        return

    # Display gauges
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig1 = create_metrics_gauge(metrics.get('avg_faithfulness', 0), "Faithfulness")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("💡 Answers grounded in sources without hallucination")

    with col2:
        fig2 = create_metrics_gauge(metrics.get('avg_relevance', 0), "Relevance")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("💡 Answers directly address questions")

    with col3:
        fig3 = create_metrics_gauge(metrics.get('avg_context_precision', 0), "Context Precision")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("💡 Retrieved sources are relevant")

    with col4:
        overall = (
            metrics.get('avg_faithfulness', 0) +
            metrics.get('avg_relevance', 0) +
            metrics.get('avg_context_precision', 0)
        ) / 3
        fig4 = create_metrics_gauge(overall, "Overall Quality")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("💡 Average of all metrics")

    st.divider()

    # Technical performance
    st.markdown("### ⚡ Technical Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Latency", f"{metrics.get('avg_latency', 0):.2f}s")
        st.caption(f"Range: {metrics.get('min_latency', 0):.2f}s - {metrics.get('max_latency', 0):.2f}s")

    with col2:
        st.metric("Total Queries", metrics.get('total_queries', 0))
        st.caption("Test set size")

    with col3:
        st.metric("Evaluation Date", metrics.get('evaluation_date', 'N/A')[:10])
        st.caption("Last system evaluation")

    # Per-topic breakdown
    if 'per_topic' in metrics and metrics['per_topic']:
        st.divider()
        st.markdown("### 🎯 Per-Topic Performance")

        topic_data = metrics['per_topic']
        topics = list(topic_data.keys())
        qualities = [topic_data[t]['avg_quality'] * 100 for t in topics]

        fig = go.Figure(data=[
            go.Bar(
                x=[t.capitalize() for t in topics],
                y=qualities,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(topics)]
            )
        ])

        fig.update_layout(
            title="Quality by Cancer Type",
            yaxis_title="Quality Score (%)",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

def display_about_tab():
    """Tab 3: About the application - Enhanced design"""

    # Hero Section (Full Width)
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🔬 Urological Oncology RAG System</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Evidence-based medical research powered by AI retrieval-augmented generation
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Two-Column Layout
    left_col, right_col = st.columns([3, 2], gap="large")

    # ========================================
    # LEFT COLUMN - Main Content
    # ========================================
    with left_col:
        # What is This?
        st.subheader("💡 What is This?")
        st.write("""
        An AI-powered research assistant that provides evidence-based answers from 815
        peer-reviewed papers across 4 urological cancer types. Uses advanced RAG
        architecture to deliver accurate, cited responses with zero hallucination.
        """)

        st.markdown("---")

        # How It Works (Visual Flow)
        st.subheader("🔄 How It Works")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>❓</div>
                <div style='font-weight: bold;'>Ask</div>
                <div style='font-size: 0.8rem; color: #666;'>Your question</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🔍</div>
                <div style='font-weight: bold;'>Search</div>
                <div style='font-size: 0.8rem; color: #666;'>41K+ chunks</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🤖</div>
                <div style='font-weight: bold;'>Generate</div>
                <div style='font-size: 0.8rem; color: #666;'>AI answer</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>📚</div>
                <div style='font-weight: bold;'>Cite</div>
                <div style='font-size: 0.8rem; color: #666;'>Source papers</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Key Features (Expandable)
        st.subheader("✨ Key Features")

        with st.expander("📚 Comprehensive Knowledge Base", expanded=False):
            st.write("""
            - **815 full-text papers** from PubMed Central Open Access
            - **4 cancer types:** Prostate, Bladder, Kidney, Testicular
            - **41,970 section-aware chunks** for precise retrieval
            - **Years covered:** 2015-2025 (latest research)
            """)

        with st.expander("🎯 Advanced RAG Pipeline", expanded=False):
            st.write("""
            **Retrieval:** Semantic search using OpenAI embeddings (text-embedding-3-small)

            **Generation:** GPT-4o-mini produces answers grounded in sources

            **Quality:** 97.5% overall score with 100% faithfulness (zero hallucination)

            **Speed:** Smart caching delivers 99.9% faster repeat queries
            """)

        with st.expander("💬 Conversation Memory", expanded=False):
            st.write("""
            - Multi-turn conversations with context awareness
            - Automatic query rewriting using conversation history
            - Maintains relevance across follow-up questions
            - Optional: enable/disable chat mode per session
            """)

        with st.expander("📊 Quality Evaluation", expanded=False):
            st.write("""
            - **Faithfulness:** Measures answer grounding in sources
            - **Relevance:** Ensures answers address the question
            - **Context Precision:** Validates retrieved source quality
            - Real-time evaluation with LLM-as-judge methodology
            """)

        with st.expander("🔗 Citation Tracking", expanded=False):
            st.write("""
            - Inline citations linked to source papers
            - Direct links to PubMed entries
            - Section-level source attribution
            - Expandable source cards with full context
            """)

        st.markdown("---")

        # Usage & Access
        st.subheader("🔑 Access & Usage")

        usage_col1, usage_col2 = st.columns(2)

        with usage_col1:
            st.markdown("""
            **🆓 Demo Mode**
            - 2 free queries per session
            - Full feature access
            - Perfect for testing
            - No sign-up required
            """)

        with usage_col2:
            st.markdown("""
            **🔐 Your API Key**
            - Unlimited queries
            - ~$0.01 per search
            - Priority access
            - [Get free key →](https://platform.openai.com/api-keys)
            """)

    # ========================================
    # RIGHT COLUMN - Stats & Tech
    # ========================================
    with right_col:
        # Quick Stats
        st.subheader("📊 Dataset Stats")

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Papers", "815", help="Full-text peer-reviewed articles")
            st.metric("Chunks", "41,970", help="Section-aware segments")
            st.metric("Avg Latency", "~7.2s", help="Query response time")

        with stat_col2:
            st.metric("Topics", "4", help="Cancer types covered")
            st.metric("Quality", "97.5%", help="Overall system quality")
            st.metric("Faithfulness", "100%", help="Zero hallucination")

        st.markdown("---")

        # Performance Metrics
        st.subheader("⚡ Performance")

        # Load actual metrics if available
        metrics = load_evaluation_metrics()
        if metrics:
            st.markdown(f"""
            <div style='background: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <div style='font-size: 0.9rem;'><strong>Faithfulness:</strong> {metrics.get('avg_faithfulness', 0):.1%}</div>
                <div style='font-size: 0.9rem;'><strong>Relevance:</strong> {metrics.get('avg_relevance', 0):.1%}</div>
                <div style='font-size: 0.9rem;'><strong>Context Precision:</strong> {metrics.get('avg_context_precision', 0):.1%}</div>
                <div style='font-size: 0.9rem; margin-top: 0.5rem; color: #666;'>Based on 12 test queries</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run evaluation to see detailed metrics")

        st.markdown("---")

        # Technology Stack
        st.subheader("🛠️ Tech Stack")

        st.markdown("""
        <div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;'>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>Python 3.11</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>OpenAI</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>GPT-4o-mini</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>ChromaDB</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>LangChain</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>BioPython</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>Streamlit</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>Plotly</span>
            <span style='background: #1f77b4; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.85rem;'>Docker</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick Links
        st.subheader("🔗 Quick Links")

        link_col1, link_col2 = st.columns(2)

        with link_col1:
            st.link_button("📂 GitHub", "https://github.com/steph-grigors/urological-oncology-rag-prod", use_container_width=True)
            st.link_button("📚 Docs", "https://github.com/steph-grigors/urological-oncology-rag-prod#readme", use_container_width=True)

        with link_col2:
            st.link_button("💼 Portfolio", "https://www.stephan-gs.work", use_container_width=True)
            st.link_button("🔗 LinkedIn", "https://linkedin.com/in/stéphan-grs", use_container_width=True)

        st.markdown("---")

        # System Status
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #e8f5e9; border-radius: 10px;'>
            <div style='font-size: 1.5rem;'>✅</div>
            <div style='font-weight: bold; color: #2e7d32;'>System Operational</div>
            <div style='font-size: 0.8rem; color: #666;'>Ready to answer your research questions!</div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================
    # BOTTOM SECTION - Data Sources & Developer
    # ========================================
    st.divider()

    # Data Sources
    st.subheader("📖 Data Sources")

    source_col1, source_col2, source_col3, source_col4 = st.columns(4)

    with source_col1:
        st.markdown("""
        **Prostate Cancer**
        - 250 papers
        - 13,541 chunks
        """)

    with source_col2:
        st.markdown("""
        **Bladder Cancer**
        - 250 papers
        - 13,152 chunks
        """)

    with source_col3:
        st.markdown("""
        **Kidney Cancer**
        - 250 papers
        - 12,777 chunks
        """)

    with source_col4:
        st.markdown("""
        **Testicular Cancer**
        - 65 papers
        - 2,500 chunks
        """)

    st.caption("All papers sourced from PubMed Central Open Access Subset")

    st.divider()

    # Developer Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h3>👨‍💻 Developed by Stéphan Grigorescu</h3>
        <p style='color: #666;'>
            Data Scientist & AI Engineer | Specializing in NLP, RAG Systems & Medical AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Social Links (centered)
    social_col1, social_col2, social_col3, social_col4 = st.columns([1, 1, 1, 1])

    with social_col1:
        st.link_button("🌐 Portfolio", "https://www.stephan-gs.work", use_container_width=True)
    with social_col2:
        st.link_button("💼 LinkedIn", "https://linkedin.com/in/stéphan-grs", use_container_width=True)
    with social_col3:
        st.link_button("🐙 GitHub", "https://github.com/steph-grigors", use_container_width=True)
    with social_col4:
        st.link_button("📧 Contact", "mailto:stephan.grigorescu@gmail.com", use_container_width=True)

    # Disclaimers
    disclaimer_col1, disclaimer_col2 = st.columns(2)

    with disclaimer_col1:
        with st.expander("⚖️ Legal Disclaimer"):
            st.caption("""
            This application is provided for educational and research purposes only.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the advice of qualified health providers with questions regarding
            medical conditions. The AI-generated answers should be verified against original
            sources before clinical application.
            """)

    with disclaimer_col2:
        with st.expander("📜 Data Attribution"):
            st.caption("""
            All research papers are sourced from PubMed Central Open Access Subset under
            Creative Commons licenses. Citations and links to original papers are provided
            for all responses. This system respects copyright and attribution requirements
            as specified by the National Library of Medicine.
            """)


def main():
    """Main Streamlit app"""

    # Compact header
    st.markdown('<div class="main-header">🔬 Urological Oncology RAG System</div>', unsafe_allow_html=True)

    # Sidebar
    top_k, model, show_context = display_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 System Performance", "ℹ️ About"])


    # Tab 1: Query Interface
    with tab1:
        # Load RAG system
        user_key = st.session_state.get('user_api_key')
        retriever = load_rag_system(_api_key=user_key)
        retriever.config.top_k = top_k

       # 1. KNOWLEDGE BASE (First)
        st.markdown("### 📚 Knowledge Base")

        # Topic selector
        topic_filter = st.selectbox(
            "Search in:",
            ["All Topics", "Prostate Cancer", "Bladder Cancer", "Kidney Cancer", "Testicular Cancer"],
            index=0,
            help="Filter by cancer type (auto-detects by default)"
        )

        # Add vertical spacing
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Conversation controls in row
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Reset Current Chat", use_container_width=True):
                memory = st.session_state.conversation_memory
                st.session_state.conversation = memory.create_conversation()
                st.session_state.current_response = None
                st.session_state.quality_metrics = None
                st.rerun()

        with col2:
            context_mode = st.checkbox(
                "💬 Enable Chat Mode",
                value=st.session_state.conversation is not None,
                help="Multi-turn conversation with context"
            )

        if context_mode and st.session_state.conversation is None:
            memory = st.session_state.conversation_memory
            st.session_state.conversation = memory.create_conversation()
        elif not context_mode:
            st.session_state.conversation = None

        # Show conversation status inline
        if st.session_state.conversation:
            turns = len(st.session_state.conversation.messages) // 2
            st.caption(f"💬 Active chat • {turns} turns")
        else:
            pass

        st.divider()

        # 3. EXAMPLE QUERIES (Third)
        st.markdown("### 💡 Example Queries")
        st.markdown("Select an example or type your own below:")

        # Add spacing
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)

        # Mixed topic examples
        example_query = st.selectbox(
            "Choose an example:",
            [
                # Prostate
                "What are the current treatment options for prostate cancer?",
                "What are the side effects of androgen deprivation therapy?",
                "What is castration-resistant prostate cancer?",
                # Bladder
                "What are the treatment options for bladder cancer?",
                "What is the role of BCG immunotherapy in bladder cancer?",
                "What are the side effects of intravesical therapy?",
                # Kidney
                "What is the role of immunotherapy in kidney cancer?",
                "What are targeted therapies for renal cell carcinoma?",
                "What are the treatment options for advanced kidney cancer?",
                # Testicular
                "What are the chemotherapy options for testicular cancer?",
                "What is the cure rate for testicular cancer?",
                "What are the side effects of chemotherapy for germ cell tumors?",
            ],
            key="example_selector",
            label_visibility="collapsed"
        )

        st.divider()

        # 4. QUERY INPUT (Fourth)
        st.markdown("### 🔍 Your Question")

        query = st.text_area(
            "Ask your question:",
            value=example_query if example_query else st.session_state.get('query', ''),
            height=120,
            placeholder="Ask about prostate cancer treatments, diagnosis, biomarkers, side effects, etc.",
            key="query_input",
            label_visibility="collapsed"
        )

        # Search buttons
        col1, col2 = st.columns([1, 3])
        with col1:
            search_button = st.button("🚀 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        if clear_button:
            st.session_state.query = ''
            st.session_state.current_response = None
            st.session_state.quality_metrics = None
            st.rerun()

        # Execute query
        if search_button and query:
            query = query.strip()

            # CHECK QUERY LIMIT
            user_has_key = st.session_state.get('user_api_key') is not None
            free_queries_used = st.session_state.query_count

            if not user_has_key and free_queries_used >= 2:
                st.error("⚠️ **Free queries limit reached!**")
                st.info("""
                You've used your 2 free queries. To continue:

                1. 🔑 Enter your OpenAI API key in the sidebar
                2. 🌐 Get a key at: https://platform.openai.com/api-keys
                3. 💰 Free tier includes $5 credit for new users
                """)
                st.stop()

            # Increment counter for free tier
            if not user_has_key:
                st.session_state.query_count += 1

            with st.spinner("🔍 Searching knowledge base..."):
                start_time = time.time()

                try:
                    # Check conversation mode
                    if st.session_state.conversation:
                        if st.session_state.conv_rag is None:
                            st.session_state.conv_rag = ConversationalRAG(
                                retriever,
                                st.session_state.conversation_memory
                            )

                        response = st.session_state.conv_rag.query(
                            question=query,
                            conversation=st.session_state.conversation,
                            model=model
                        )

                        rewritten_query = response.get('rewritten_query')
                    else:
                        response = retriever.query(
                            question=query,
                            model=model,
                            return_sources=True,
                            use_cache=True
                        )
                        rewritten_query = None

                    latency = time.time() - start_time

                    # Store response in session state
                    st.session_state.current_response = {
                        'query': query,
                        'answer': response['answer'],
                        'sources': response.get('sources', []),
                        'num_sources': response['num_sources'],
                        'latency': latency,
                        'rewritten_query': rewritten_query
                    }

                    # Clear previous quality metrics
                    st.session_state.quality_metrics = None

                    # Track session metrics
                    st.session_state.session_metrics['queries'].append(query)
                    st.session_state.session_metrics['latencies'].append(latency)
                    if latency < 0.5:
                        st.session_state.session_metrics['cache_hits'] += 1

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())

        elif search_button:
            st.warning("⚠️ Please enter a question")

        # DISPLAY RESULTS (separate from search button, uses session state)
        if st.session_state.current_response:
            resp = st.session_state.current_response

            st.divider()

            # Show query rewrite if conversation mode
            if resp['rewritten_query'] and resp['rewritten_query'] != resp['query']:
                with st.expander("🔄 Query Rewrite (Context Applied)", expanded=True):
                    st.markdown(f"**Original:** {resp['query']}")
                    st.markdown(f"**Expanded:** {resp['rewritten_query']}")

            # Display results header
            is_cached = resp['latency'] < 0.5
            cache_indicator = " ✨ (cached)" if is_cached else ""

            st.success(f"✅ Found {resp['num_sources']} relevant sources in {resp['latency']:.2f}s{cache_indicator}")

            st.divider()

            # Answer with inline citations
            st.markdown("### 📄 Answer")

            formatted_answer = format_answer_with_citations(
                resp['answer'],
                resp['sources']
            )
            st.markdown(formatted_answer, unsafe_allow_html=True)


             # Quality evaluation button - evaluates and switches to Metrics tab
            # Quality evaluation button - displays inline
            if st.button("🔬 Evaluate Response Quality"):
                with st.spinner("🔬 Evaluating response quality..."):
                    metrics = evaluate_response_quality(
                        retriever,
                        resp['query'],
                        resp['answer'],
                        resp['sources']
                    )
                    st.session_state.quality_metrics = metrics

            # Display inline quality metrics if available
            if st.session_state.quality_metrics:
                st.markdown("#### 📊 Quality Metrics for This Response")
                m = st.session_state.quality_metrics

                col1, col2, col3 = st.columns(3)

                with col1:
                    fig1 = create_metrics_gauge(m['faithfulness'], "Faithfulness")
                    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                    st.caption("Is answer grounded in sources?")

                with col2:
                    fig2 = create_metrics_gauge(m['relevance'], "Relevance")
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
                    st.caption("Does answer address question?")

                with col3:
                    fig3 = create_metrics_gauge(m['precision'], "Context Precision")
                    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
                    st.caption("Are sources relevant?")

                overall = (m['faithfulness'] + m['relevance'] + m['precision']) / 3

                if overall >= 0.9:
                    st.success(f"✅ Excellent quality: {overall:.1%}")
                elif overall >= 0.8:
                    st.info(f"✅ Good quality: {overall:.1%}")
                else:
                    st.warning(f"⚠️ Quality score: {overall:.1%}")

            st.divider()

            # Sources
            display_sources(resp['sources'], show_context)

    # Tab 2: Metrics Dashboard
    with tab2:
        display_metrics_dashboard()

    # Tab 3: About
    with tab3:
        display_about_tab()

if __name__ == "__main__":
    main()
