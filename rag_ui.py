"""
Streamlit Web Interface — Urological Oncology RAG System
Clinical Dashboard v2
"""

import streamlit as st
import os
import time
import uuid
import requests
import plotly.graph_objects as go

# ── Backend API configuration ─────────────────────────────────────────────────
_API_BASE = os.environ.get("API_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY  = os.environ.get("API_KEY", "")


@st.cache_resource
def _get_session() -> requests.Session:
    """One pooled, keep-alive HTTP session per Streamlit process.

    A plain module-level `requests.Session()` would be recreated on every
    rerun (Streamlit re-executes the whole script top to bottom on each
    interaction) — st.cache_resource is what actually persists it.
    """
    return requests.Session()

_QUALITY_BADGES: dict[str, tuple[str, str, str]] = {
    "high":         ("#1e7e34", "🟢 High Evidence",
                     "Multiple consistent, relevant sources were retrieved. "
                     "The answer is well-supported by the knowledge base."),
    "hedged":       ("#d97706", "🟡 Hedged",
                     "Relevant sources were found but the evidence is mixed or limited. "
                     "The answer reflects uncertainty in the literature."),
    "caveated":     ("#dc2626", "🔴 Caveated",
                     "Sources were retrieved but may not directly address the query, "
                     "or the evidence conflicts. Interpret with caution."),
    "insufficient": ("#6b7280", "⚫ Insufficient",
                     "No sufficiently relevant literature was found in the knowledge base. "
                     "The response draws on the model's general medical knowledge, "
                     "not peer-reviewed sources in this database."),
}

_DESIGN_COLORS: dict[str, str] = {
    "rct":           "#1966D3",
    "meta_analysis": "#0e4fa8",
    "cohort":        "#059669",
    "case_report":   "#7c3aed",
    "review":        "#9333ea",
    "unknown":       "#9ca3af",
}

_CANCER_TYPES = [
    "All Topics",
    "Prostate Cancer",
    "Bladder Cancer",
    "Kidney Cancer",
    "Testicular Cancer",
    "Penile Cancer",
    "Adrenal Cancer",
]

# /treatment-card requires a single specific cancer_type — "All Topics" isn't valid.
_CARD_CANCER_TYPES = _CANCER_TYPES[1:]

st.set_page_config(
    page_title="Urological Oncology RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Layout ── */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        max-width: 270px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.25rem !important;
    }
    /* ── Sidebar branding ── */
    .sb-brand {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1966D3;
        letter-spacing: -0.2px;
    }
    .sb-tagline {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 0.1rem;
        margin-bottom: 1rem;
    }
    .sb-section {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9ca3af;
        margin-bottom: 0.4rem;
        margin-top: 0.1rem;
    }
    /* ── Sidebar input placeholder ── */
    section[data-testid="stSidebar"] input::placeholder {
        font-size: 0.78rem;
    }
    /* ── Answer container ── */
    .answer-body {
        line-height: 1.8;
        font-size: 0.97rem;
    }
    /* ── Citation chip ── */
    .citation {
        display: inline-block;
        background: #dbeafe;
        color: #1966D3;
        font-weight: 600;
        font-size: 0.8em;
        padding: 1px 5px;
        border-radius: 4px;
    }
    /* ── Study design badge ── */
    .design-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        color: white;
        letter-spacing: 0.03em;
    }
    /* ── Evidence quality pill ── */
    .ev-pill {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 11px;
        border-radius: 12px;
        color: white;
        cursor: help;
    }
    /* ── Status bar ── */
    .status-bar {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }
    /* ── Settings panel heading ── */
    .panel-heading {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }
    /* ── Thin rule ── */
    .thin-rule {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    /* ── Tab gap ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    /* ── Button full width ── */
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "conversation_id"    not in st.session_state: st.session_state.conversation_id    = None
if "current_response"   not in st.session_state: st.session_state.current_response   = None
if "quality_metrics"    not in st.session_state: st.session_state.quality_metrics    = None
if "quality_history"    not in st.session_state: st.session_state.quality_history    = []
if "query_count"        not in st.session_state: st.session_state.query_count        = 0
if "last_latency"       not in st.session_state: st.session_state.last_latency       = None
if "avg_latency"        not in st.session_state: st.session_state.avg_latency        = None
if "latency_sum"        not in st.session_state: st.session_state.latency_sum        = 0.0
if "top_k"              not in st.session_state: st.session_state.top_k              = 5
if "show_context"       not in st.session_state: st.session_state.show_context       = False
if "custom_system_prompt" not in st.session_state: st.session_state.custom_system_prompt = ""
if "tc_custom_system_prompt" not in st.session_state: st.session_state.tc_custom_system_prompt = ""
if "user_api_key"       not in st.session_state: st.session_state.user_api_key       = None
if "chat_mode"          not in st.session_state: st.session_state.chat_mode          = False
if "endpoint_mode"      not in st.session_state: st.session_state.endpoint_mode      = "query"
if "tc_patient_id"      not in st.session_state: st.session_state.tc_patient_id      = str(uuid.uuid4())[:8]
if "tc_current_response" not in st.session_state: st.session_state.tc_current_response = None
if "tc_input_method"    not in st.session_state: st.session_state.tc_input_method    = "structured"


# ── Backend helpers ────────────────────────────────────────────────────────────

def _query_backend(
    query: str,
    cancer_types: list,
    top_k: int,
    conversation_id: str | None,
    system_prompt: str | None = None,
) -> dict:
    api_key = st.session_state.get("user_api_key") or _API_KEY
    headers = {"X-API-Key": api_key} if api_key else {}
    payload: dict = {"query": query, "cancer_types": cancer_types, "top_k": top_k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if system_prompt:
        payload["system_prompt"] = system_prompt
    resp = _get_session().post(f"{_API_BASE}/query", json=payload, headers=headers, timeout=180)
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}", response=resp)
    return resp.json()


def _treatment_card_backend(
    patient_id: str,
    cancer_type: str,
    age_range: str,
    clinical_history: str,
    comorbidities: dict,
    top_k: int,
    system_prompt: str | None = None,
) -> dict:
    """Call /treatment-card. Always requests English output with citations kept
    in the drug/sources fields (so they can be rendered as chips against
    sources_detail) and the parametric-fallback disclosure enabled — this UI is
    the only caller that opts into either, onco-review-app's notebook does not."""
    api_key = st.session_state.get("user_api_key") or _API_KEY
    headers = {"X-API-Key": api_key} if api_key else {}
    payload: dict = {
        "patient_id": patient_id,
        "cancer_type": cancer_type,
        "age_range": age_range,
        "clinical_history": clinical_history,
        "comorbidities": comorbidities,
        "top_k": top_k,
        "language": "en",
        "keep_citations": True,
        "disclose_fallback": True,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    resp = _get_session().post(f"{_API_BASE}/treatment-card", json=payload, headers=headers, timeout=180)
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {detail}", response=resp)
    return resp.json()


def _parse_comorbidities(text: str) -> dict:
    """Parse a 'Name: value' per-line text block into a dict."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key:
            result[key] = value or "yes"
    return result


def _design_badge(design: str) -> str:
    color = _DESIGN_COLORS.get(design, "#9ca3af")
    label = design.replace("_", " ").title()
    return f"<span class='design-badge' style='background:{color};'>{label}</span>"


def _ev_pill(quality: str) -> str:
    entry = _QUALITY_BADGES.get(quality)
    if entry:
        color, label, tooltip = entry
    else:
        color, label, tooltip = "#6b7280", quality.replace("_", " ").title(), ""
    return f"<span class='ev-pill' style='background:{color};' title='{tooltip}'>{label}</span>"


def _format_citations(answer: str, sources: list) -> str:
    for i, source in enumerate(sources):
        tag = f"[Doc {i + 1}]"
        title = (source.get("title") or "")[:60]
        if tag in answer:
            answer = answer.replace(tag, f'<span class="citation" title="{title}">{tag}</span>')
    return answer


def _evaluate_quality(query: str, answer: str, sources: list) -> dict:
    from src.evaluation.judges import JudgeSet

    class _Chunk:
        def __init__(self, text: str, metadata: dict):
            self.text = text
            self.metadata = metadata

    chunks = [
        _Chunk(s.get("key_finding", ""), {"evidence_level": 2, "study_design": s.get("study_design", "unknown")})
        for s in sources
    ]
    scores = JudgeSet().score_all(question=query, answer=answer, chunks=chunks)
    return {"faithfulness": scores.faithfulness, "relevance": scores.answer_relevance, "precision": scores.context_precision}


# ── Sidebar ────────────────────────────────────────────────────────────────────

def display_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='sb-section'>🔑 Access</div>", unsafe_allow_html=True)
        user_api_key = st.text_input(
            "Access key",
            type="password",
            placeholder="Provided by administrator",
            label_visibility="collapsed",
            key="api_key_input",
        )
        if user_api_key:
            st.session_state.user_api_key = user_api_key
            st.success("✅ Key active")
        else:
            st.session_state.user_api_key = None
            st.caption("Contact your administrator for a key.")

        with st.expander("🔒 Security", expanded=False):
            st.caption("• Keys managed per user")
            st.caption("• Rate limiting enforced server-side")
            st.caption("• All queries logged for quality monitoring")

        st.divider()

        st.markdown("<div class='sb-section'>📊 Session Info</div>", unsafe_allow_html=True)
        if st.session_state.conversation_id:
            cid = st.session_state.conversation_id[:8] + "…"
            st.markdown(
                f"<div style='font-size:0.75rem;color:#6b7280;margin-bottom:0.1rem;'>Session ID</div>"
                f"<div style='font-size:1.1rem;font-weight:600;color:#059669;font-family:monospace;margin-bottom:0.6rem;'>{cid}</div>",
                unsafe_allow_html=True,
            )
        c1, c2 = st.columns(2)
        with c1:
            last_lat = st.session_state.last_latency
            st.metric("Last query", f"{last_lat:.1f}s" if last_lat is not None else "—")
        with c2:
            avg_lat = st.session_state.avg_latency
            st.metric("Avg latency", f"{avg_lat:.1f}s" if avg_lat is not None else "—")
        st.metric("Queries this session", st.session_state.query_count)



# ── Source cards ───────────────────────────────────────────────────────────────

def display_sources(sources: list, show_context: bool) -> None:
    st.markdown(f"### 📚 Sources ({len(sources)})")
    for idx, source in enumerate(sources, 1):
        title      = (source.get("title")  or "Unknown")[:80]
        year       = source.get("year", "")
        design     = source.get("study_design", "unknown")
        sample     = source.get("sample_size")
        pmid       = source.get("pmid", "")
        authors    = source.get("authors", "")
        journal    = source.get("journal", "")
        key_finding = source.get("key_finding", "")
        section    = source.get("section", "")

        with st.container(border=True):
            t_col, y_col = st.columns([6, 1])
            with t_col:
                st.markdown(f"**[{idx}] {title}**")
            with y_col:
                if year:
                    st.markdown(
                        f"<div style='text-align:right;color:#9ca3af;font-size:0.85rem;'>{year}</div>",
                        unsafe_allow_html=True,
                    )

            if authors or journal:
                auth_str = f"{authors[:65]}{'…' if len(authors) > 65 else ''}" if authors else ""
                jour_str = f" · *{journal}*" if journal else ""
                st.caption(f"{auth_str}{jour_str}")

            meta: list[str] = [_design_badge(design)]
            if sample:
                n_str = f"{sample:,}" if isinstance(sample, int) else str(sample)
                meta.append(f"<span style='font-size:0.78rem;color:#6b7280;'>N = {n_str}</span>")
            if section:
                meta.append(f"<span style='font-size:0.78rem;color:#6b7280;'>§ {section.title()}</span>")
            if pmid:
                meta.append(
                    f"<a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}' target='_blank' "
                    f"style='font-size:0.78rem;color:#1966D3;text-decoration:none;'>PMID {pmid} ↗</a>"
                )
            st.markdown(
                "<div style='display:flex;gap:0.65rem;align-items:center;flex-wrap:wrap;margin:0.25rem 0;'>"
                + " ".join(meta) + "</div>",
                unsafe_allow_html=True,
            )

            if key_finding:
                st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)
                preview = key_finding if show_context else key_finding[:220] + ("…" if len(key_finding) > 220 else "")
                st.caption(preview)


# ── Treatment card ───────────────────────────────────────────────────────────────

def display_treatment_card(card: dict, show_context: bool) -> None:
    retrieval_meta = card.get("retrieval_metadata", {}) or {}
    sources_detail = retrieval_meta.get("sources_detail", []) or []

    if retrieval_meta.get("grounded") is False:
        st.warning(
            "⚠ No relevant literature was retrieved for this case. The "
            "recommendations below are based on the model's general clinical "
            "knowledge, not the indexed evidence base — verify before clinical use."
        )

    st.markdown(f"### 🧾 Treatment Card — Patient {card.get('patient_id', '')}")

    # `confidence`/`guideline` stay short in practice (single word/code) so a
    # KPI-style st.metric works. `treatment_confidence` is often a full
    # justification sentence — st.metric truncates long values with no
    # wrapping, so it's rendered as wrapped text below instead.
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence", card.get("confidence", "—"))
    with c2:
        st.metric("Guideline", card.get("guideline", "—"))

    if card.get("stage"):
        st.caption(f"**Stage:** {card['stage']}")
    if card.get("comorbidities_impact"):
        st.caption(f"**Comorbidities impact:** {card['comorbidities_impact']}")
    if card.get("treatment_confidence"):
        st.caption(f"**Treatment confidence:** {card['treatment_confidence']}")

    st.divider()
    st.markdown("#### 💊 Recommended treatments")
    for t in card.get("treatment", []):
        drug_html = _format_citations(t.get("drug", ""), sources_detail)
        level = t.get("level", "")
        intent = t.get("intent", "")
        with st.container(border=True):
            st.markdown(f"<div class='answer-body'>{drug_html}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='design-badge' style='background:#1966D3;'>{intent}</span> "
                f"<span class='design-badge' style='background:#6b7280;margin-left:0.4rem;'>"
                f"Level {level}</span>",
                unsafe_allow_html=True,
            )
            for w in t.get("warnings", []):
                st.error(f"**{w.get('type', '').title()} warning** — {w.get('message', '')}")

    st.divider()
    if sources_detail:
        display_sources(sources_detail, show_context)
    elif card.get("sources"):
        st.markdown(f"### 📚 Sources ({len(card['sources'])})")
        for s in card["sources"]:
            st.caption(s)


# ── Query tab ──────────────────────────────────────────────────────────────────

def display_query_tab() -> None:
    st.markdown("<div class='panel-heading'>🔀 Mode</div>", unsafe_allow_html=True)
    mode_label = st.radio(
        "Mode",
        ["💬 Clinical Query", "🧾 Treatment Card"],
        index=0 if st.session_state.endpoint_mode == "query" else 1,
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Clinical Query: free-text question answered with inline [Doc N] citations. "
            "Treatment Card: structured per-patient recommendation (stage, treatment "
            "options, evidence level) generated via the dedicated /treatment-card endpoint — "
            "prompts, citation handling, and fallback disclosure all switch automatically."
        ),
    )
    new_mode = "query" if mode_label.endswith("Clinical Query") else "treatment_card"
    if new_mode != st.session_state.endpoint_mode:
        st.session_state.endpoint_mode = new_mode
        st.rerun()

    if st.session_state.endpoint_mode == "query":
        _display_clinical_query_mode()
    else:
        _display_treatment_card_mode()


def _display_clinical_query_mode() -> None:
    col_main, col_settings = st.columns([3, 1], gap="large")

    # ── Right column: settings (always visible) ────────────────────────────
    with col_settings:
        st.markdown("<div class='panel-heading'>⚙️ Query Settings</div>", unsafe_allow_html=True)

        topic_filter = st.selectbox(
            "Cancer type",
            _CANCER_TYPES,
            index=0,
            help=(
                "Restrict answers to a single cancer type. "
                "Use 'All Topics' to search across the entire knowledge base — "
                "useful for questions that span multiple cancer types or involve shared treatments."
            ),
        )
        st.session_state.top_k = st.slider(
            "Sources retrieved",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help=(
                "Number of source excerpts used to build the answer. "
                "Higher values bring in more evidence but may include less directly relevant passages. "
                "5 is a good default for most clinical questions."
            ),
        )
        st.session_state.show_context = st.checkbox(
            "Show full source text",
            value=st.session_state.show_context,
        )

        st.markdown("<div class='panel-heading' style='margin-top:0.75rem;'>💬 Chat Mode</div>", unsafe_allow_html=True)
        chat_mode = st.toggle(
            "Enable Chat Mode",
            value=st.session_state.chat_mode,
            key="chat_toggle",
            help=(
                "Enables multi-turn conversation mode: the assistant maintains full context "
                "across follow-up questions, so you can refine, compare, or dig deeper "
                "without repeating yourself."
            ),
        )
        if chat_mode != st.session_state.chat_mode:
            st.session_state.chat_mode = chat_mode
            if chat_mode and st.session_state.conversation_id is None:
                st.session_state.conversation_id = str(uuid.uuid4())
            elif not chat_mode:
                st.session_state.conversation_id = None

        if st.session_state.conversation_id:
            if st.button("↺ New conversation", use_container_width=True):
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.current_response = None
                st.session_state.quality_metrics  = None
                st.session_state.quality_history  = []
                st.session_state.query_count      = 0
                st.session_state.last_latency     = None
                st.session_state.avg_latency      = None
                st.session_state.latency_sum      = 0.0
                st.rerun()

    # ── Left column: query input + results ────────────────────────────────
    with col_main:
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:600;margin-bottom:0.6rem;'>"
            "Hi! I specialise in urological oncology. What's your clinical question?"
            "</div>",
            unsafe_allow_html=True,
        )

        query = st.text_area(
            "Query",
            value=st.session_state.get("query_val", ""),
            height=130,
            placeholder=(
                "e.g. What is the evidence for enzalutamide in metastatic hormone-sensitive "
                "prostate cancer?"
            ),
            key="query_input",
            label_visibility="collapsed",
        )

        b1, b2, _ = st.columns([2, 2, 6])
        with b1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with b2:
            clear_button = st.button("✕ Clear", use_container_width=True)

        if clear_button:
            st.session_state["query_val"] = ""
            st.session_state["query_input"] = ""
            st.session_state.current_response = None
            st.session_state.quality_metrics = None
            st.rerun()

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        with st.expander("📝 Custom Instructions (System prompt)", expanded=True):
            st.session_state.custom_system_prompt = st.text_area(
                "Override system prompt",
                value=st.session_state.custom_system_prompt,
                height=110,
                placeholder=(
                    "Optional. Tell the assistant how to tailor its answers — e.g. "
                    "'Focus on first-line treatment options only', "
                    "'Always include survival statistics where available', or "
                    "'Summarise findings in plain language suitable for a patient consultation'."
                ),
                label_visibility="collapsed",
            )

        if search_button and query:
            query = query.strip()
            with st.spinner("Searching augmented knowledge database…"):
                start = time.time()
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
                        system_prompt=st.session_state.custom_system_prompt or None,
                    )
                    latency_ms = response.get("latency_ms", {})
                    latency = latency_ms.get("total", (time.time() - start) * 1000) / 1000

                    st.session_state.current_response = {
                        "query":            query,
                        "answer":           response.get("answer", ""),
                        "sources":          response.get("sources", []),
                        "num_sources":      len(response.get("sources", [])),
                        "latency":          latency,
                        "evidence_quality": response.get("evidence_quality", "insufficient"),
                        "confidence_score": response.get("confidence_score", 0.0),
                    }

                    try:
                        metrics = _evaluate_quality(query, response.get("answer", ""), response.get("sources", []))
                        st.session_state.quality_metrics = metrics
                        st.session_state.quality_history.append({**metrics, "query": query})
                    except Exception:
                        st.session_state.quality_metrics = None

                    st.session_state.query_count  += 1
                    st.session_state.last_latency  = latency
                    st.session_state.latency_sum  += latency
                    st.session_state.avg_latency   = st.session_state.latency_sum / st.session_state.query_count

                except Exception as e:
                    import traceback
                    short = str(e).split(":")[0] if ":" in str(e) else str(e)
                    st.error(f"Query failed — {short}")
                    with st.expander("Debug info", expanded=False):
                        st.code(str(e))
                        st.code(traceback.format_exc())

        elif search_button:
            st.warning("Please enter a question.")

        # ── Results ───────────────────────────────────────────────────────
        if st.session_state.current_response:
            resp = st.session_state.current_response
            st.divider()

            ev   = resp.get("evidence_quality", "insufficient")
            pill = _ev_pill(ev)
            st.markdown(
                f"<div class='status-bar'>"
                f"Found <strong>{resp['num_sources']} sources</strong> · "
                f"{resp['latency']:.1f}s · {pill}"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown(_format_citations(resp["answer"], resp["sources"]), unsafe_allow_html=True)

            st.divider()
            display_sources(resp["sources"], st.session_state.show_context)


def _display_treatment_card_mode() -> None:
    col_main, col_settings = st.columns([3, 1], gap="large")

    # ── Right column: settings ──────────────────────────────────────────────
    with col_settings:
        st.markdown("<div class='panel-heading'>⚙️ Card Settings</div>", unsafe_allow_html=True)

        cancer_type_label = st.selectbox(
            "Cancer type",
            _CARD_CANCER_TYPES,
            index=0,
            help="Treatment cards require one specific cancer type — unlike Clinical Query, there is no 'All Topics' option.",
        )
        st.session_state.top_k = st.slider(
            "Sources retrieved",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of retrieved source excerpts used to build the card.",
        )
        st.session_state.show_context = st.checkbox(
            "Show full source text",
            value=st.session_state.show_context,
        )

        if st.button("↺ New patient", use_container_width=True):
            st.session_state.tc_patient_id = str(uuid.uuid4())[:8]
            st.session_state.tc_current_response = None
            st.rerun()

    # ── Left column: patient form + results ─────────────────────────────────
    with col_main:
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:600;margin-bottom:0.6rem;'>"
            "Generate a structured treatment card from a clinical case."
            "</div>",
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns([1, 1])
        with p1:
            patient_id = st.text_input("Patient ID", value=st.session_state.tc_patient_id)
        with p2:
            age_range = st.text_input("Age range (optional)", placeholder="e.g. 70-79")

        input_method_label = st.radio(
            "Input method",
            ["📋 Structured form", "📝 Free text"],
            index=0 if st.session_state.tc_input_method == "structured" else 1,
            horizontal=True,
            help=(
                "Structured form: separate fields for clinical history and comorbidities. "
                "Free text: paste the whole case as one block (history + comorbidities together) — "
                "useful when copying a case from elsewhere."
            ),
        )
        st.session_state.tc_input_method = (
            "structured" if input_method_label.endswith("Structured form") else "free_text"
        )

        if st.session_state.tc_input_method == "structured":
            clinical_history = st.text_area(
                "Clinical history",
                height=130,
                placeholder=(
                    "e.g. Metastatic hormone-sensitive prostate cancer, cT3b N1 M1b, "
                    "ISUP grade 4, PSA 42 ng/mL. No prior systemic treatment."
                ),
            )
            comorbidities_text = st.text_area(
                "Comorbidities (optional — one per line, 'Name: value')",
                height=80,
                placeholder="Chronic kidney disease: stage 3\nDiabetes: type 2",
            )
        else:
            clinical_history = st.text_area(
                "Clinical case (free text)",
                height=200,
                placeholder=(
                    "e.g. Metastatic hormone-sensitive prostate cancer, cT3b N1 M1b, "
                    "ISUP grade 4, PSA 42 ng/mL. No prior systemic treatment.\n\n"
                    "Comorbidities: chronic kidney disease stage 3, type 2 diabetes."
                ),
            )
            comorbidities_text = ""  # folded into clinical_history above

        b1, b2, _ = st.columns([2, 2, 6])
        with b1:
            generate_button = st.button("🧾 Generate Card", type="primary", use_container_width=True)
        with b2:
            clear_button = st.button("✕ Clear", use_container_width=True, key="tc_clear")

        if clear_button:
            st.session_state.tc_current_response = None
            st.rerun()

        with st.expander("📝 Custom Instructions (System prompt)", expanded=False):
            st.session_state.tc_custom_system_prompt = st.text_area(
                "Override system prompt",
                value=st.session_state.tc_custom_system_prompt,
                height=110,
                placeholder="Optional. Overrides the default card-generation instructions entirely.",
                label_visibility="collapsed",
                key="tc_system_prompt",
            )

        if generate_button and clinical_history.strip() and len(clinical_history.strip()) >= 10:
            cancer_type = cancer_type_label.lower().replace(" cancer", "").strip()
            with st.spinner("Generating treatment card…"):
                try:
                    card = _treatment_card_backend(
                        patient_id=patient_id or st.session_state.tc_patient_id,
                        cancer_type=cancer_type,
                        age_range=age_range,
                        clinical_history=clinical_history.strip(),
                        comorbidities=_parse_comorbidities(comorbidities_text),
                        top_k=st.session_state.top_k,
                        system_prompt=st.session_state.tc_custom_system_prompt or None,
                    )
                    st.session_state.tc_current_response = card
                except Exception as e:
                    import traceback
                    short = str(e).split(":")[0] if ":" in str(e) else str(e)
                    st.error(f"Card generation failed — {short}")
                    with st.expander("Debug info", expanded=False):
                        st.code(str(e))
                        st.code(traceback.format_exc())
        elif generate_button:
            st.warning("Please enter a clinical history (at least 10 characters).")

        if st.session_state.tc_current_response:
            st.divider()
            display_treatment_card(st.session_state.tc_current_response, st.session_state.show_context)


# ── System Performance tab ─────────────────────────────────────────────────────

def display_performance_tab() -> None:
    st.markdown("## 📊 System Performance")
    metrics = st.session_state.quality_metrics
    history = st.session_state.quality_history

    if not metrics:
        st.info("Run a query to see quality metrics for each response.")
        return

    overall = (metrics["faithfulness"] + metrics["relevance"] + metrics["precision"]) / 3

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        delta_color = "normal" if overall >= 0.8 else "inverse"
        st.metric("Overall", f"{overall:.0%}")
    with m2:
        st.metric("Faithfulness", f"{metrics['faithfulness']:.0%}")
    with m3:
        st.metric("Relevance", f"{metrics['relevance']:.0%}")
    with m4:
        st.metric("Context Precision", f"{metrics['precision']:.0%}")

    st.divider()
    col_bars, col_explain = st.columns([2, 3], gap="large")

    with col_bars:
        st.markdown("#### Score breakdown")
        for label, val in [
            ("Faithfulness", metrics["faithfulness"]),
            ("Relevance",    metrics["relevance"]),
            ("Context Precision", metrics["precision"]),
        ]:
            st.markdown(f"**{label}** — {val:.0%}")
            st.progress(val)

    with col_explain:
        st.markdown("#### What these scores mean")
        with st.expander("**Faithfulness** — Is the answer grounded in sources?", expanded=True):
            st.caption(
                "Measures whether every factual claim in the answer is traceable to a [Doc N] citation. "
                "Penalises unsupported clinical directives, missing citations, and numeric claims not present in any retrieved chunk."
            )
        with st.expander("**Relevance** — Does the answer address the question?", expanded=True):
            st.caption(
                "Measures keyword overlap between the query and answer, length appropriateness, "
                "and whether the answer directly responds rather than restating the question."
            )
        with st.expander("**Context Precision** — Are the retrieved sources on-topic?", expanded=True):
            st.caption(
                "Evaluates keyword overlap between retrieved chunks and the query, source diversity "
                "(multiple independent papers > single source), and section relevance (Results/Conclusion > Methods)."
            )

    if len(history) >= 1:
        st.divider()
        st.markdown("#### Overall quality score over this session")
        x_labels = [f"Q{i}" for i in range(1, len(history) + 1)]
        overall_scores = [
            (h["faithfulness"] + h["relevance"] + h["precision"]) / 3 * 100
            for h in history
        ]
        hover_text = [
            f"Q{i}: {h['query'][:50]}{'…' if len(h['query']) > 50 else ''}<br>Overall: {overall_scores[i-1]:.0f}%<br>Faithfulness: {h['faithfulness']:.0%}<br>Relevance: {h['relevance']:.0%}<br>Precision: {h['precision']:.0%}"
            for i, h in enumerate(history, 1)
        ]
        fig = go.Figure()
        fig.add_hrect(
            y0=80, y1=100,
            fillcolor="#059669", opacity=0.07,
            layer="below", line_width=0,
        )
        fig.add_hline(
            y=80,
            line=dict(color="#059669", width=1.5, dash="dot"),
            annotation_text="80% target",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#059669"),
        )
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=overall_scores,
            mode="lines+markers",
            line=dict(color="#1966D3", width=3),
            marker=dict(size=9, color="#1966D3", line=dict(color="white", width=2)),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 100], ticksuffix="%", tickfont=dict(size=11)),
            xaxis=dict(tickfont=dict(size=11)),
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── About tab ──────────────────────────────────────────────────────────────────

def display_about_tab() -> None:
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.subheader("✨ Key Features")

        with st.expander("📚 Comprehensive Knowledge Base", expanded=False):
            st.write("""
            - **31,000+ full-text papers** from PubMed Central Open Access
            - **6 cancer types:** Prostate, Bladder, Kidney, Testicular, Penile, Adrenal
            - **795,000+ section-aware chunks** for precise retrieval
            - **Years covered:** 2010–2026 (latest high-evidence research)
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

    with right_col:
        st.subheader("📊 Dataset at a Glance")
        for _label, _value in [
            ("Papers", "31,000+"),
            ("Chunks", "795,000+"),
            ("Topics", "6 cancer types"),
            ("Evidence filter", "RCT+"),
            ("Avg latency", "~35s"),
            ("Years", "2010–2026"),
        ]:
            _c1, _c2 = st.columns(2)
            with _c1:
                st.caption(_label)
            with _c2:
                st.markdown(f"**{_value}**")

    st.divider()
    ds_col, ts_col = st.columns([3, 2], gap="large")
    with ds_col:
        st.subheader("📖 Data Sources")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        with c1:
            st.markdown("**Prostate Cancer**\n- 17,382 papers\n- 445,895 chunks")
        with c2:
            st.markdown("**Bladder Cancer**\n- 5,476 papers\n- 139,933 chunks")
        with c3:
            st.markdown("**Kidney Cancer**\n- 6,034 papers\n- 152,113 chunks")
        with c4:
            st.markdown("**Testicular Cancer**\n- 782 papers\n- 18,479 chunks")
        with c5:
            st.markdown("**Adrenal Cancer**\n- 1,384 papers\n- 31,774 chunks")
        with c6:
            st.markdown("**Penile Cancer**\n- 303 papers\n- 7,112 chunks")
        st.caption("All papers sourced from PubMed Central Open Access Subset")
    with ts_col:
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
    lc1, lc2 = st.columns(2)
    with lc1:
        st.link_button("📂 GitHub", "https://github.com/steph-grigors/urological-oncology-rag-prod", use_container_width=True)
        st.link_button("💼 Portfolio", "https://www.stephan-gs.work", use_container_width=True)
    with lc2:
        st.link_button("🔗 LinkedIn", "https://linkedin.com/in/stéphan-grs", use_container_width=True)
        st.link_button("📧 Contact", "mailto:stephan.grigorescu@gmail.com", use_container_width=True)

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
    <div style='border-left:4px solid #1966D3;padding:0.4rem 1rem;margin-bottom:1.25rem;'>
        <div style='font-size:1.5rem;font-weight:800;letter-spacing:-0.3px;'>
            🔬 Urological Oncology RAG
        </div>
        <div style='font-size:0.85rem;color:#6b7280;margin-top:0.15rem;'>
            Evidence-based clinical AI ·
            <strong style='color:#1966D3;'>31,000+</strong> papers ·
            <strong style='color:#1966D3;'>795K+</strong> chunks ·
            <strong style='color:#1966D3;'>6</strong> cancer types ·
            <strong style='color:#1966D3;'>2010–2026</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 Performance", "ℹ️ About"])

    with tab1:
        display_query_tab()
    with tab2:
        display_performance_tab()
    with tab3:
        display_about_tab()


if __name__ == "__main__":
    main()
