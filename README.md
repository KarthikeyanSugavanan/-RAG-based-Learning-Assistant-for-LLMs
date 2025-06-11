# RAG-based Learning Assistant for LLMs

🚀 **Advanced RAG System with RARE Framework for Enhanced Domain Knowledge Retrieval**

## 📖 Overview

The RAG-based Learning Assistant is a sophisticated Retrieval-Augmented Generation (RAG) system designed to address the limitations of traditional Large Language Models (LLMs) in domain-specific question answering. By combining external document retrieval with powerful language models, this system significantly reduces hallucinations and improves factual accuracy in specialized domains.

## 🎯 Key Features

### 🔧 Advanced Architecture
- **Hybrid Retrieval System**: FAISS for semantic similarity + DuckDB for SQL-powered metadata filtering
- **RARE Framework**: Retrieval-Augmented Reasoning and Generation for structured multi-step reasoning
- **Multi-Model Support**: LLaMA-2 7B-Chat, GPT-3.5, and Zephyr-7B-Alpha implementations
- **Cross-Encoder Reranking**: Enhanced relevance scoring with ms-marco-MiniLM-L-6-v2

### 📊 Performance Metrics
- **38-42% Reduction** in hallucinations across all models
- **22% Greater Accuracy** improvement for multi-hop questions
- **50.5 ROUGE-L Score** - nearly matching human-level performance
- **45.1 BLEU Score** with LLaMA-2 + RARE + FAISS configuration
- **29× Better Cost Efficiency** compared to GPT-3.5 API

## 🏗️ System Architecture

### Core Components

1. **Embedding Layer**
   - Base: BAAI/bge-base-en (1024-dimension embeddings)
   - Domain-adapted variants for Scientific, Legal, and Technical domains
   - Dynamic embedding selection based on query classification

2. **Retrieval Pipeline**
   - **FAISS Vector Store**: GPU-accelerated similarity search with IVF/HNSW indexing
   - **DuckDB Integration**: SQL capabilities for metadata-aware filtering
   - **Hierarchical Retrieval**: Document chunks → full documents for balanced recall/precision

3. **RARE Reasoning Framework**
   - Explicit information synthesis from multiple sources
   - Focused question formulation for relevance maintenance
   - Confidence estimation and contradiction resolution

4. **Generation Models**
   | Model | Parameters | Context Window | Quantization | Specialization |
   |-------|------------|----------------|--------------|----------------|
   | LLaMA-2 7B-Chat | 7B | 4K tokens | 4-bit (5.2GB) | Conversational QA |
   | GPT-3.5 | ~175B | 8K tokens | N/A | General Knowledge |
   | Zephyr-7B-Alpha | 7B | 32K tokens | 4-bit AWQ (4.8GB) | Efficient Inference |

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch transformers
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install duckdb
pip install langchain
pip install sentence-transformers

# Additional requirements
pip install numpy pandas matplotlib seaborn
pip install rouge-score nltk
pip install huggingface-hub
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs.git
   cd -RAG-based-Learning-Assistant-for-LLMs
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**
   ```bash
   # Download embedding model
   python scripts/download_embeddings.py
   
   # Download LLaMA-2 (requires permission)
   python scripts/setup_llama.py
   ```

### Quick Start

```python
from rag_assistant import RAGLearningAssistant
from rag_assistant.models import LlamaModel
from rag_assistant.retrieval import HybridRetriever

# Initialize the system
retriever = HybridRetriever(
    faiss_index_path="data/faiss_index",
    duckdb_path="data/documents.db"
)

model = LlamaModel(
    model_path="models/llama-2-7b-chat",
    use_rare_prompting=True
)

rag_system = RAGLearningAssistant(retriever=retriever, model=model)

# Ask a question
question = "What are the key differences between supervised and unsupervised learning?"
answer = rag_system.generate_answer(question)
print(answer)
```

## 📁 Project Structure

```
RAG-based-Learning-Assistant-for-LLMs/
├── src/
│   ├── rag_assistant/
│   │   ├── models/           # LLM implementations
│   │   ├── retrieval/        # FAISS + DuckDB retrieval
│   │   ├── prompting/        # RARE framework
│   │   └── evaluation/       # Metrics and benchmarking
│   ├── data/
│   │   ├── documents/        # Knowledge base
│   │   ├── faiss_index/      # Vector embeddings
│   │   └── processed/        # Preprocessed data
│   ├── notebooks/
│   │   ├── evaluation.ipynb  # Performance analysis
│   │   └── examples.ipynb    # Usage examples
│   └── scripts/
│       ├── build_index.py    # Index construction
│       └── evaluate.py       # Benchmark evaluation
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── requirements.txt
```

## 🔬 Methodology

### 1. Document Processing
- Chunk documents into semantically coherent passages
- Generate embeddings using BAAI/bge-base-en
- Store in FAISS index with DuckDB metadata

### 2. Query Processing
- Embed user question into vector space
- Retrieve top-k relevant documents using hybrid search
- Rerank passages using cross-encoder model

### 3. RARE Generation
- **Retrieve**: Gather relevant context from multiple sources
- **Analyze**: Synthesize information and identify key points
- **Reason**: Generate intermediate questions and reasoning steps
- **Execute**: Produce final answer with source attribution

## 📊 Evaluation Results

### Quantitative Performance

| System | BLEU Score | ROUGE-L | Hallucination Reduction |
|--------|------------|---------|------------------------|
| LLaMA-2 + RARE + FAISS | **45.1** | **50.5** | **40%** |
| GPT-3.5 + FAISS | 43.5 | 48.9 | 35% |
| Zephyr + FAISS | 39.2 | 44.3 | 38% |
| Baseline (No RAG) | 35.7 | 42.1 | - |

### Key Findings
- **Multi-hop questions** show 22% greater improvement compared to single-hop queries
- **Cost efficiency**: Local LLaMA-2 provides 29× better queries/$ than GPT-3.5 API
- **Latency consistency**: p95 latency remains <2s even for complex 5-hop questions

## 🔧 Configuration

### Model Configuration

```python
# config.yaml
model:
  name: "llama-2-7b-chat"
  quantization: "4-bit"
  max_tokens: 512
  temperature: 0.3
  
retrieval:
  faiss:
    index_type: "IVF"
    nprobe: 32
    gpu_enabled: true
  duckdb:
    enable_fts: true
    metadata_filtering: true
    
rare:
  enable_reasoning: true
  max_reasoning_steps: 5
  confidence_threshold: 0.7
```

## 🎯 Use Cases

### Healthcare
- Medical literature synthesis
- Drug interaction queries
- Clinical decision support

### Legal
- Case law research
- Regulatory compliance
- Contract analysis

### Education
- Academic research assistance
- Curriculum development
- Student tutoring systems

### Technical Documentation
- API documentation queries
- Troubleshooting guides
- Best practices retrieval

## 🚀 Advanced Features

### Custom Domain Adaptation
```python
# Fine-tune embeddings for specific domains
from rag_assistant.adaptation import DomainAdapter

adapter = DomainAdapter(base_model="bge-base-en")
adapter.fine_tune(
    domain_corpus="medical_papers.jsonl",
    output_path="models/medical-bge"
)
```

### Multi-Language Support
```python
# Configure for multiple languages
config = {
    "embeddings": {
        "english": "BAAI/bge-base-en",
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }
}
```

## 🔬 Research Paper

This implementation is based on our research paper: **"Questron: Retrieval-Augmented Generation for Domain-Specific Question Answer Generation"** (DS 5983 - Large Language Models Project Report)

**Authors**: Kameswara Sai Srikar Manda, Karthikeyan Sugavanan, Pramukh Venkatesh Koushik

## 📈 Future Roadmap

- [ ] **Scaling**: Support for massive corpora (10M+ documents)
- [ ] **Dynamic Retrieval**: Query complexity-based retrieval strategies
- [ ] **Multi-Modal**: Image and video content integration
- [ ] **Real-time Learning**: Continuous knowledge base updates
- [ ] **Federated RAG**: Distributed knowledge retrieval across organizations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs.git
cd -RAG-based-Learning-Assistant-for-LLMs
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{rag_learning_assistant2025,
    title={RAG-based Learning Assistant for LLMs: Retrieval-Augmented Generation for Domain-Specific Question Answer Generation},
    author={Manda, Kameswara Sai Srikar and Sugavanan, Karthikeyan and Koushik, Pramukh Venkatesh},
    year={2025},
    institution={DS 5983 - Large Language Models},
    url={https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs}
}
```

## 🔗 Links

- **GitHub Repository**: [https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs](https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs)
- **Research Paper**: [Available in /docs](docs/rag_learning_assistant_paper.pdf)
- **Demo**: [Interactive Demo](https://rag-assistant-demo.streamlit.app)

## 📞 Contact

For questions or collaboration opportunities:

- **Karthikeyan Sugavanan**: [GitHub Profile](https://github.com/KarthikeyanSugavanan)
- **Project Issues**: [GitHub Issues](https://github.com/KarthikeyanSugavanan/-RAG-based-Learning-Assistant-for-LLMs/issues)

---

**Built with ❤️ for advancing domain-specific AI knowledge retrieval**
