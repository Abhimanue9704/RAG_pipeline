# 📄 RAG Pipeline – Document Assistant using LangChain & FAISS

A Retrieval-Augmented Generation (RAG) style document assistant built using LangChain, FAISS, and HuggingFace models.  
This project ingests PDF and text documents, cleans and chunks them intelligently, builds a FAISS vector index, retrieves relevant chunks for user queries, and reranks results using a cross-encoder for improved accuracy.

---

## 🚀 Features

- Supports PDF and TXT documents
- Advanced OCR and text cleaning
- Sentence-aware and token-based chunking
- Semantic embeddings using HuggingFace
- Persistent FAISS vector index
- Cross-encoder reranking for higher precision
- Metadata-based filtering (source and section)
- Configurable retrieval and reranking thresholds
- Outputs structured JSON results

---

## 🗂️ Project Structure

```
testing-langchain/
├── doc_assistant.py
├── loaders/
│   └── pdf_loader.py
├── input_docs/
├── queries/
│   └── questions.txt
├── data/
│   └── faiss_index/
├── output/
│   └── result.json
├── requirements.txt
├── requirements.lock.txt
├── .gitignore
└── README.md
```

---

## 🧠 How the Pipeline Works

1. **Document Loading**
   - Loads TXT files directly
   - Loads PDF files using a custom PDF loader
   - Cleans OCR artifacts, boilerplate text, and formatting noise

2. **Chunking**
   - Sentence-level splitting using SpaCy
   - Token-based chunking using TokenTextSplitter
   - Each chunk is enriched with metadata including:
     - Source document
     - Document ID
     - Chunk ID and index
     - Token estimate
     - Detected section

3. **Vector Indexing**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
   - Builds a FAISS vector index
   - Persists the index locally for reuse

4. **Retrieval**
   - Performs semantic similarity search
   - Supports filtering by source document and section
   - Supports FAISS similarity score thresholding

5. **Reranking**
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Reranks retrieved chunks for higher accuracy

6. **Output**
   - Saves results as structured JSON including content, metadata, and rerank scores

---

## 🛠️ Installation

### Create Virtual Environment

```bash
python -m venv env
source env/bin/activate   # Linux / macOS
env\Scripts\activate      # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## ▶️ Usage

### Basic Command

```bash
python doc_assistant.py \
  --docs input_docs \
  --queries queries/questions.txt \
  --output output/result.json
```

---

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
|--------|-------------|---------|
| --docs | Path to documents directory | Required |
| --queries | Path to queries file | Required |
| --output | Output JSON file | Required |
| --data | Data directory | data |
| --reindex | Rebuild FAISS index | False |
| --top_k | Initial retrieval count | 10 |
| --final_k | Final results after reranking | 3 |
| --filter_source | Filter by source document | None |
| --filter_section | Filter by section name | None |
| --score_threshold | FAISS similarity threshold | 0.0 |
| --chunk_size | Token chunk size | 350 |
| --chunk_overlap | Token overlap | 80 |
| --min_rerank_score | Minimum rerank score | -5.0 |

---

## 📄 Example Output (JSON)

```json
[
  {
    "query": "What are the machine guarding requirements?",
    "results": [
      {
        "content": "Machine guards must prevent contact with moving parts...",
        "metadata": {
          "source": "Safety Committee Handout.pdf",
          "section": "MACHINE GUARDING",
          "doc_id": 0,
          "chunk_id": 12,
          "chunk_index": 3
        },
        "rerank_score": 7.42
      }
    ],
    "total_retrieved": 10,
    "total_reranked": 3
  }
]
```

---

## ⚠️ Notes

- Do NOT commit the virtual environment (`env/`)
- FAISS indexes can be regenerated and are optional to commit
- Ensure `.gitignore` excludes:
  - env/
  - __pycache__/
  - *.pyc

---

## 📌 Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Transformers
- SentenceTransformers
- SpaCy

---

## 📜 License

This project is provided for educational and internal use.
