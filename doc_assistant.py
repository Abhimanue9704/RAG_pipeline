from pathlib import Path
import argparse
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import SpacyTextSplitter, TokenTextSplitter
from loaders.pdf_loader import load_pdf
from collections import defaultdict
from langchain_community.vectorstores import FAISS
import os
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder



def build_faiss_index(chunks, embeddings, index_path: Path):
    """
    Build and persist FAISS index from chunks
    """
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    return vectorstore

def load_faiss_index(index_path: Path, embeddings):
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_queries(queries_path: Path):
    """
    Load queries from a text file (one question per line)
    """
    with open(queries_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Document Assistant using LangChain + FAISS"
    )
    parser.add_argument("--docs", required=True, help="Path to documents directory")
    parser.add_argument("--queries", required=True, help="Path to questions file")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--data", default="data", help="Path to data directory")
    parser.add_argument("--reindex", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--top_k", type=int, default=10, help="Top K chunks to retrieve (before reranking)")
    parser.add_argument("--final_k", type=int, default=3, help="Final K chunks after reranking")
    parser.add_argument("--filter_source", type=str, default=None, help="Filter by source document")
    parser.add_argument("--filter_section", type=str, default=None, help="Filter by section")
    parser.add_argument("--score_threshold", type=float, default=0.0, help="Minimum similarity score threshold")
    parser.add_argument("--chunk_size", type=int, default=350, help="Token chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=80, help="Token chunk overlap")
    parser.add_argument("--min_rerank_score", type=float, default=-5.0, help="Minimum reranking score threshold")
    return parser.parse_args()

# Add section-aware splitting
def split_on_sections(text: str):
    """Split text on clear section headers"""
    section_pattern = r'\n\n([A-Z][A-Za-z\s]+:)\n'
    sections = re.split(section_pattern, text)
    # Reconstruct sections with their headers
    result = []
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            result.append(sections[i] + '\n' + sections[i+1])
    return result

def clean_text(text: str) -> str:
    # 1. Fix common OCR spacing issues (targeted)
    replacements = {
        r"\bfi ve\b": "five",
        r"\bopera ng\b": "operating",
        r"\bopera on\b": "operation",
        r"\bEjecon\b": "Ejection",
        r"\bSafegaurding\b": "Safeguarding",
        r"\bself-adjusng\b": "self-adjusting",
        r"\bmo ons\b": "motions",
        r"\bcon nuous\b": "continuous",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 2. Remove OCR artifacts
    text = re.sub(r"\(cid:\d+\)", "", text)

    # 3. Fix line-break word splits
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 4. Remove boilerplate
    boilerplate_patterns = [
        r"Company Name:.*",
        r"Date:.*",
        r"Address,.*",
        r"\(\d{3}\)\s*\d{3}-\d{4}",
        r"www\.\S+",
        r"_{5,}",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 5. Remove numbering-only lines
    text = re.sub(r"\n\s*\d+\.\s*\n", "\n", text)

    # 6. Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def detect_section(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()

        if not (5 <= len(line) <= 80):
            continue

        if (
            line.isupper()
            or re.match(r"^[A-Z][A-Za-z\s\-]{3,50}:$", line)
        ):
            return line

    return "general"



def load_documents(docs_path: Path):
    documents = []

    for file_path in docs_path.iterdir():
        if file_path.suffix.lower() == ".txt":
            raw_text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=clean_text(raw_text),
                    metadata={"source": file_path.name, "method": "text"},
                )
            )

        elif file_path.suffix.lower() == ".pdf":
            pdf_docs = load_pdf(file_path)
            for doc in pdf_docs:
                doc.page_content = clean_text(doc.page_content)
                documents.append(doc)

    return documents



def chunk_documents(documents, chunk_size=350, chunk_overlap=80):
    sentence_splitter = SpacyTextSplitter(
        pipeline="en_core_web_sm",
    )

    token_splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 1. Split documents
    sentence_docs = sentence_splitter.split_documents(documents)
    chunks = token_splitter.split_documents(sentence_docs)

    # 2. Group chunks by source (IMPORTANT FIX)
    chunks_by_source = defaultdict(list)
    for c in chunks:
        chunks_by_source[c.metadata["source"]].append(c)

    enriched_chunks = []
    global_chunk_id = 0

    # 3. Assign metadata efficiently
    for doc_index, doc in enumerate(documents):
        source = doc.metadata["source"]
        doc_chunks = chunks_by_source.get(source, [])

        for local_index, chunk in enumerate(doc_chunks):
            chunk.metadata.update({
                "doc_id": doc_index,
                "chunk_id": global_chunk_id,
                "chunk_index": local_index,
                "char_count": len(chunk.page_content),
                "token_estimate": len(chunk.page_content) // 4,
                "section": detect_section(chunk.page_content),
            })

            enriched_chunks.append(chunk)
            global_chunk_id += 1

    return enriched_chunks


def apply_metadata_filters(results, filter_source=None, filter_section=None):
    """
    Filter results based on metadata criteria
    """
    filtered = results
    
    if filter_source:
        filtered = [r for r in filtered if r.metadata.get("source") == filter_source]
    
    if filter_section:
        filtered = [r for r in filtered if filter_section.lower() in r.metadata.get("section", "").lower()]
    
    return filtered


def rerank_results(query: str, results, reranker, top_k=3, min_score=-5.0):
    """
    Rerank results using a cross-encoder model and filter by score
    """
    if not results:
        return []
    
    pairs = [[query, doc.page_content] for doc in results]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(results, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by minimum score AND top_k
    filtered = [(doc, float(score)) for doc, score in scored_docs 
                if score >= min_score][:top_k]
    
    return filtered

def main():
    args = parse_args()
    docs_path = Path(args.docs)
    index_path = Path(args.data) / "faiss_index"

    print("Loading documents...")
    documents = load_documents(docs_path)

    print(f"Chunking documents (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print(f"Total chunks: {len(chunks)}")

    # 1️⃣ Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2️⃣ Initialize reranker
    print("Loading reranker model...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 3️⃣ Build or load FAISS
    if args.reindex or not (index_path / "index.faiss").exists():
        print("Building FAISS index...")
        vectorstore = build_faiss_index(chunks, embeddings, index_path)
    else:
        print("Loading existing FAISS index...")
        vectorstore = load_faiss_index(index_path, embeddings)

    print("FAISS index ready ✅")

    queries_path = Path(args.queries)

    print("Loading queries...")
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")

    all_results = []

    for q in queries:
        # print(f"\n{'='*60}")
        # print(f"🔎 Query: {q}")
        # print(f"{'='*60}")
        
        # Retrieve initial candidates with scores
        results_with_scores = vectorstore.similarity_search_with_score(q, k=args.top_k)
        
        # Extract documents and scores
        results = [doc for doc, score in results_with_scores]
        initial_scores = [float(score) for doc, score in results_with_scores]
        
        # print(f"📊 Retrieved {len(results)} initial candidates")
        
        # Apply metadata filters
        if args.filter_source or args.filter_section:
            results = apply_metadata_filters(
                results, 
                filter_source=args.filter_source,
                filter_section=args.filter_section
            )
            # print(f"🔍 After filtering: {len(results)} chunks")
        
        # Apply score threshold
        if args.score_threshold > 0.0:
            filtered_results = []
            for doc, score in zip(results, initial_scores[:len(results)]):
                if score <= args.score_threshold:  # Lower score = better for FAISS
                    filtered_results.append(doc)
            results = filtered_results
            # print(f"📉 After score threshold: {len(results)} chunks")
        
        # Rerank results
        if results:
            # print(f"🎯 Reranking top {min(len(results), args.final_k)} results...")
            reranked_results = rerank_results(q, results, reranker, top_k=args.final_k)
        else:
            reranked_results = []

        query_results = []
        for idx, (doc, rerank_score) in enumerate(reranked_results, 1):
            # print(f"\n--- Result {idx} (Rerank Score: {rerank_score:.4f}) ---")
            # print(f"Source: {doc.metadata.get('source')}")
            # print(f"Section: {doc.metadata.get('section')}")
            # print(f"Chunk: {doc.metadata.get('chunk_index')}/{doc.metadata.get('chunk_id')}")
            # print(f"\nContent Preview:")
            # print(doc.page_content[:300])
            # print("...")

            query_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": rerank_score
            })

        all_results.append({
            "query": q,
            "results": query_results,
            "total_retrieved": len(results),
            "total_reranked": len(reranked_results)
        })

    # Write results to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()