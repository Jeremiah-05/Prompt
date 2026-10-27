"""
RAG Retriever — TF-IDF cosine similarity over chunked policy.md and faq.md.
No external vector DB needed; uses scikit-learn's TfidfVectorizer.
"""

import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"

# Module-level cache for vectorizer and document matrix
_vectorizer = None
_tfidf_matrix = None
_chunks = []


def _chunk_markdown(filepath: Path, chunk_size: int = 200) -> list[dict]:
    """Split a markdown file into chunks of approximately chunk_size words.
    
    Splits on section headers (## ...) first, then further splits if a section
    exceeds chunk_size words.
    
    Args:
        filepath: Path to markdown file
        chunk_size: Target words per chunk
    
    Returns:
        List of dicts with 'text', 'source', and 'section' keys.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    source = filepath.name
    chunks = []
    
    # Split by section headers (## ...)
    sections = re.split(r'\n(?=##?\s)', content)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Extract section title
        lines = section.split("\n")
        title = lines[0].strip("#").strip() if lines[0].startswith("#") else "General"
        text = section
        
        words = text.split()
        if len(words) <= chunk_size:
            chunks.append({
                "text": text,
                "source": source,
                "section": title,
            })
        else:
            # Split into sub-chunks
            for i in range(0, len(words), chunk_size):
                sub_text = " ".join(words[i:i + chunk_size])
                chunks.append({
                    "text": sub_text,
                    "source": source,
                    "section": f"{title} (part {i // chunk_size + 1})",
                })
    
    return chunks


def _build_index():
    """Build TF-IDF index from policy.md and faq.md chunks."""
    global _vectorizer, _tfidf_matrix, _chunks
    
    if _vectorizer is not None:
        return  # Already built
    
    all_chunks = []
    
    # Chunk policy.md
    policy_path = DATA_DIR / "policy.md"
    if policy_path.exists():
        all_chunks.extend(_chunk_markdown(policy_path, chunk_size=200))
    
    # Chunk faq.md
    faq_path = DATA_DIR / "faq.md"
    if faq_path.exists():
        all_chunks.extend(_chunk_markdown(faq_path, chunk_size=150))
    
    if not all_chunks:
        _chunks = []
        return
    
    _chunks = all_chunks
    texts = [c["text"] for c in all_chunks]
    
    _vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams + bigrams for better matching
    )
    _tfidf_matrix = _vectorizer.fit_transform(texts)


def retrieve_policy_chunks(query: str, top_k: int = 3, threshold: float = 0.05) -> list[dict]:
    """Retrieve the most relevant policy/FAQ chunks for a query.
    
    Args:
        query: User's query string
        top_k: Number of top chunks to return
        threshold: Minimum cosine similarity score to include
    
    Returns:
        List of dicts with 'text', 'source', 'section', and 'score' keys,
        sorted by descending relevance score.
    """
    _build_index()
    
    if not _chunks or _vectorizer is None:
        return []
    
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    
    # Get top_k indices sorted by score
    ranked_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in ranked_indices:
        score = float(similarities[idx])
        if score >= threshold:
            results.append({
                "text": _chunks[idx]["text"],
                "source": _chunks[idx]["source"],
                "section": _chunks[idx]["section"],
                "score": score,
            })
    
    return results


def format_rag_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a string for prompt injection.
    
    Args:
        chunks: List of chunk dicts from retrieve_policy_chunks
    
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    lines = ["Retrieved policy/FAQ context:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{i}] (source: {chunk['source']}, section: {chunk['section']}, relevance: {chunk['score']:.2f})")
        lines.append(chunk["text"])
    
    return "\n".join(lines)


def reset_index():
    """Reset the cached index (useful for testing)."""
    global _vectorizer, _tfidf_matrix, _chunks
    _vectorizer = None
    _tfidf_matrix = None
    _chunks = []
