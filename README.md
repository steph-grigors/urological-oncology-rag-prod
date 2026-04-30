# 🔬 Urological Oncology RAG System

A production-grade Retrieval-Augmented Generation (RAG) system providing evidence-based answers about urological oncology.

## 🎯 Live Demo

Try it now: [Hugging Face Space](https://huggingface.co/spaces/steph-grigors/urological-oncology-rag)

## 📊 System Overview

- **Papers:** 815 full-text peer-reviewed articles from PubMed Central
- **Topics:** Prostate, Bladder, Kidney, and Testicular Cancer
- **Chunks:** 41,970 section-aware searchable segmentrag_u
- **Coverage:** 2015-2025 (last 10 years)
- **Average Sections:** 15.5 per paper
- **Quality Score:** 95.8% (Faithfulness + Relevance + Precision)

## ✨ Key Features

### 🎯 Multi-Topic Coverage
- **Prostate Cancer** (250 papers, 13,541 chunks)
- **Bladder Cancer** (250 papers, 13,152 chunks)
- **Kidney Cancer** (250 papers, 12,712 chunks)
- **Testicular Cancer** (65 papers, 2,565 chunks)

### 🔍 Advanced Retrieval
- ✅ **Section-aware chunking** - Preserves document structure
- ✅ **Semantic search** - OpenAI embeddings (1536 dimensions)
- ✅ **Top-K retrieval** - Configurable source count
- ✅ **Citation tracking** - Every answer links to sources

### 🧠 Intelligent Features
- ✅ **Conversation memory** - Multi-turn context understanding
- ✅ **Query rewriting** - Expands vague follow-ups with context
- ✅ **Smart caching** - 99.9% speedup for repeated queries
- ✅ **Quality evaluation** - On-demand response scoring

### 🎨 User Experience
- ✅ **Clean interface** - Professional Streamlit UI
- ✅ **Example queries** - Pre-built questions per topic
- ✅ **Source preview** - Expandable citation cards
- ✅ **Real-time metrics** - Session statistics

## 🏗️ Technical Architecture

### Data Pipeline
1. **Collection:** PubMed Central API → 815 full-text XML papers
2. **Parsing:** Extract sections (Introduction, Methods, Results, Discussion, etc.)
3. **Chunking:** 200 words/chunk with 30-word overlap (section-aware)
4. **Embedding:** OpenAI text-embedding-3-small (1536 dimensions)
5. **Storage:** ChromaDB vector database (41,970 vectors)
6. **Retrieval:** Semantic similarity search (cosine distance)
7. **Generation:** GPT-4o-mini with source citations

### Technology Stack
- **Backend:** Python 3.11
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** text-embedding-3-small
- **Vector DB:** ChromaDB (persistent)
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces

## 📈 Performance Metrics

### Quality (from evaluation on test set)
- **Faithfulness:**       95.8%
- **Relevance:**          100.0%
- **Context Precision:**  91.7%
- **Overall Quality:** 95.8%

### Speed
- **Average Latency:** 5.95s (first query)
- **Cached Query:** 0.02s (99.9% faster)
- **Retrieval:** ~0.5s
- **Generation:** ~6-7s

### Scale
- **9.4x more chunks** than baseline (41,970 vs 4,459)
- **16.6x more papers** than baseline (815 vs 49)
- **4x topic coverage** (multi-cancer vs single)

## 🎓 Use Cases

### Clinical Decision Support
- Treatment option comparison
- Side effect analysis
- Drug interaction queries
- Clinical guideline lookup

### Medical Research
- Literature review automation
- Evidence synthesis
- Gap analysis
- Citation discovery

### Medical Education
- Student Q&A
- Case study research
- Exam preparation
- Continuing education

## 💡 Example Queries

### Prostate Cancer
```
"What are the current treatment options for prostate cancer?"
"What are the side effects of androgen deprivation therapy?"
"What is castration-resistant prostate cancer?"
```

### Bladder Cancer
```
"What are the treatment options for bladder cancer?"
"What is the role of BCG immunotherapy?"
"What are the side effects of intravesical therapy?"
```

### Kidney Cancer
```
"What is the role of immunotherapy in kidney cancer?"
"What are targeted therapies for renal cell carcinoma?"
```

### Testicular Cancer
```
"What are the chemotherapy options for testicular cancer?"
"What is the cure rate for testicular cancer?"
```

## 🔒 Privacy & Compliance

- ✅ No user data stored or logged
- ✅ HIPAA-compliant architecture
- ✅ Queries processed in real-time only
- ✅ No PHI collection or transmission
- ✅ OpenAI API calls encrypted (TLS)
- ✅ Open-source codebase for transparency

## 🚀 Local Deployment

### Prerequisites
- Python 3.11+
- OpenAI API key
- Docker (optional)

### Quick Start
```bash
# Clone repository
git clone https://github.com/steph-grigors/urological-oncology-rag.git
cd urological-oncology-rag

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key_here"

# Run application
streamlit run rag_ui.py
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8502
```

## 📊 Project Statistics

### Data Collection
- **Time:** 3 hours (with NCBI API key)
- **Source:** PubMed Central Open Access
- **Method:** BioPython + XML parsing

### Processing
- **Time:** 10 minutes
- **Method:** Section-aware chunking
- **Output:** 41,970 chunks

### Embeddings
- **Time:** 15 minutes
- **Cost:** $0.17 (OpenAI API)
- **Model:** text-embedding-3-small

### Total Build Time
- **End-to-End:** ~3.5 hours
- **Total Cost:** < $1.00

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional cancer types
- More papers per topic
- Multilingual support
- Advanced filtering
- Custom evaluation metrics

## 📄 Citation

If you use this system in research, please cite:
```bibtex
@software{urological_oncology_rag_2025,
  title = {Urological Oncology RAG System},
  author = {Stéphan Grigorescu},
  year = {2025},
  note = {815 papers, 41,970 chunks, PubMed Central Open Access},
  url = {https://github.com//urological-oncology-rag}
}
```

## 📧 Contact
steph-grigors
- **GitHub:** [@steph-grigors](https://github.com/steph-grigors)
- **Issues:** [Report bugs](https://github.com/steph-grigors/urological-oncology-rag/issues)

## ⚖️ License

MIT License - Free for research, educational, and commercial use.

## 🙏 Acknowledgments

- **PubMed Central** for open-access medical literature
- **OpenAI** for embedding and generation models
- **Hugging Face** for deployment platform
- **ChromaDB** for vector storage

---
