import logging
import math
import re

def cosine_similarity(vecA, vecB):
    if len(vecA) != len(vecB):
        raise ValueError("Vectors must have the same length")
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    magnitudeA = math.sqrt(sum(a * a for a in vecA))
    magnitudeB = math.sqrt(sum(b * b for b in vecB))
    return dot_product / (magnitudeA * magnitudeB) if magnitudeA > 0 and magnitudeB > 0 else 0

def jaccard_rank(query, documents):
    logging.warning(f"[fallback] Using Jaccard similarity for {len(documents)} documents")
    # Tokenize (lowercase and split on non-alphanumeric)
    query_tokens = set(re.findall(r'\w+', query.lower()))
    results = []
    for idx, doc in enumerate(documents):
        doc_tokens = set(re.findall(r'\w+', doc.lower()))
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        score = len(intersection) / len(union) if len(union) > 0 else 0
        results.append({"index": idx, "relevance_score": score})
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results}
