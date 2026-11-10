# Advanced RAG Application

A Python-based Retrieval-Augmented Generation (RAG) project built with LangChain, Sentence Transformers, and ChromaDB. The project provides the foundation for document embedding, persistent vector storage, and semantic retrieval.

> **Project status:** Work in progress. Document retrieval is implemented; document-ingestion orchestration and LLM-powered answer generation are planned improvements.

## Features

- **Semantic embeddings:** Generate text embeddings using `all-MiniLM-L6-v2`.
- **Persistent vector storage:** Store document content, metadata, and embeddings in ChromaDB.
- **Semantic retrieval:** Retrieve the top-K most relevant document chunks for a query.
- **Document management foundation:** Add document objects and their embeddings to a persistent collection.
- **Extensible architecture:** Organized into embedding, vector-storage, and retrieval components.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Core programming language |
| LangChain | Document abstractions and RAG components |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Persistent vector database |
| NumPy | Numerical operations |
| PyMuPDF / PyPDF | PDF loading dependencies |
| uv | Python dependency and environment management |
| Groq / Google Gemini integrations | Dependencies available for future LLM integration |

## Architecture

```text
              User Query
                   |
                   v
          Embedding Manager
                   |
                   v
          Sentence Transformer
             all-MiniLM-L6-v2
                   |
                   v
             Query Embedding
                   |
                   v
          ChromaDB Vector Store
                   |
                   v
           Top-K Retrieval
                   |
                   v
          Retrieved Documents
```

### Core Components

**1. EmbeddingManager**

Uses Sentence Transformers to convert text into numerical embeddings.

**2. VectorStore**

Uses ChromaDB's persistent client to store document content, metadata, and embeddings.

**3. RagRetriever**

Converts a query into an embedding and retrieves the top-K matching documents from ChromaDB.

## Project Structure

```text
Advanced-RAG-Application/
│
├── data/
│   └── vector_store/          # Persistent vector database
│
├── notebook/
│   └── document.ipynb         # Experimentation notebook
│
├── src/
│   └── rag/
│       ├── __init__.py
│       └── main.py            # Embedding, storage and retrieval
│
├── .env.example               # Environment variable template
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
├── requirement.txt
└── uv.lock
```

## Getting Started

### Prerequisites

- Python 3.12 or later
- Git
- uv

### 1. Clone the repository

```bash
git clone https://github.com/ShubhCoder-17/Advanced-RAG-Application.git

cd Advanced-RAG-Application
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the application

Run the package entry point:

```bash
uv run rag
```

To run the RAG module directly:

```bash
uv run python -m rag.main
```

The current main module prints an initialization message. The complete document-ingestion and question-answering workflow is still under development.

## Usage

The core classes can be imported into other Python modules:

```python
from rag.main import EmbeddingManager, VectorStore, RagRetriever

# Initialize the embedding manager
embedding_manager = EmbeddingManager()

# Initialize the persistent vector store
vector_store = VectorStore()

# Initialize the retriever
retriever = RagRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)

# Retrieve relevant documents
results = retriever.retrieve(
    query="What information is available in the documents?",
    top_k=5
)

for result in results:
    print(result["document"])
    print(result["metadata"])
    print(result["score"])
```

**Note:** This example assumes documents have already been inserted into the ChromaDB collection. Document loading, splitting, and embedding orchestration still need to be connected.

## Environment Variables

The current retrieval implementation does not require an LLM API key to initialize.

Groq and Google Gemini integrations are included as dependencies or imports, but neither is currently used to generate answers.

If you integrate an LLM later, store its API key in `.env` and never commit that file.

## Roadmap

- [x] Sentence Transformer embedding generation
- [x] Persistent ChromaDB vector storage
- [x] Top-K semantic retrieval
- [ ] PDF document ingestion pipeline
- [ ] Recursive text chunking
- [ ] End-to-end document processing
- [ ] Groq or Gemini LLM integration
- [ ] Context-aware answer generation
- [ ] Retrieval evaluation and relevance scoring
- [ ] Streamlit or FastAPI interface
- [ ] Automated tests

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Author

**ShubhCoder-17**

GitHub: [@ShubhCoder-17](https://github.com/ShubhCoder-17)

Repository: [Advanced-RAG-Application](https://github.com/ShubhCoder-17/Advanced-RAG-Application)
