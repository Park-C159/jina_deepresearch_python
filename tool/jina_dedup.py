import logging

from tool.cosin import cosine_similarity
from tool.embedding import get_embeddings

SIMILARITY_THRESHOLD = 0.86


# 假定你有 get_embeddings 和 cosine_similarity 的实现
# def get_embeddings(texts: list[str], tracker=None) -> list[list[float]]
# def cosine_similarity(vec1: list[float], vec2: list[float]) -> float

async def dedup_queries(new_queries, existing_queries, tracker=None):
    try:
        # 快速返回：只有一个新query且无已有query
        if len(new_queries) == 1 and len(existing_queries) == 0:
            return {"unique_queries": new_queries}

        # 批量获得所有embeddings
        all_queries = new_queries + existing_queries
        all_embeddings = get_embeddings(all_queries, tracker).get("embeddings")

        # 如果embedding结果为空（如API异常），直接返回所有新query
        if not all_embeddings or len(all_embeddings) == 0:
            return {"unique_queries": new_queries}

        # 拆分新的和已有的embeddings
        new_embeddings = all_embeddings[:len(new_queries)]
        existing_embeddings = all_embeddings[len(new_queries):]

        unique_queries = []
        used_indices = set()

        for i, query in enumerate(new_queries):
            is_unique = True

            # 与已有query语义去重
            for j, _ in enumerate(existing_queries):
                similarity = cosine_similarity(new_embeddings[i], existing_embeddings[j])
                if similarity >= SIMILARITY_THRESHOLD:
                    is_unique = False
                    break

            # 与已经判定为unique的新query再比一次
            if is_unique:
                for used_index in used_indices:
                    similarity = cosine_similarity(new_embeddings[i], new_embeddings[used_index])
                    if similarity >= SIMILARITY_THRESHOLD:
                        is_unique = False
                        break

            # 如果是“语义新”，则加入最终集合
            if is_unique:
                unique_queries.append(query)
                used_indices.add(i)

        logging.info({"unique_queries": unique_queries})
        return {"unique_queries": unique_queries}

    except Exception as e:
        logging.error(f"Error in deduplication analysis: {e}")
        return {"unique_queries": new_queries}

