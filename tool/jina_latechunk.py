# cherry_pick.py
from __future__ import annotations
import os
import math
import logging
from typing import List

import numpy as np
from dotenv import load_dotenv

from tool.cosin import cosine_similarity
from tool.embedding import get_embeddings
from utils.action_tracker import ActionTracker
from utils.schemas import LANGUAGE_CODE
from utils.token_tracker import TokenTracker

# ========== 可插拔的日志 & 埋点 ==========
load_dotenv()
log_error = logging.getLogger("cherry_pick").error
log_debug = logging.getLogger("cherry_pick").debug


class TrackerContext:
    def __init__(self):
        self.tokenTracker = TokenTracker()
        self.actionTracker = ActionTracker()

# ========== embedding 客户端 ==========
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
    raise RuntimeError("JINA_API_KEY not set")


def trim_symbols(s: str) -> str:
    # 与 TS 侧保持一致：去掉首尾空白 + 常见符号
    return s.strip(" \t\r\n。，、！？：；“”‘’\"'()[]")


# ========== 主函数 ==========
async def cherry_pick(
        query: str,
        long_context: str,
        options=None,
        trackers=None,
        url: str = "",
) -> str:
    """
    返回按相似度挑选后的 <snippet-i> 片段拼接字符串。
    任何异常都走降级：返回头部 snippetLength * numSnippets 字符。
    """
    options = options or {}
    trackers = trackers or TrackerContext()

    snippet_len = int(options.get("snippetLength", 6000))
    chunk_size = int(options.get("chunkSize", 300))
    num_snippets = int(
        options.get("numSnippets")
        or max(2, min(5, math.floor(len(long_context) / snippet_len)))
    )

    if len(long_context) < snippet_len * 2:
        log_debug("content is too short, dont bother")
        return long_context

    # 1. 生成 chunks
    chunks: List[str] = []
    for i in range(0, len(long_context), chunk_size):
        seg = long_context[i: i + chunk_size]
        if trim_symbols(seg):
            chunks.append(seg)
    log_debug(f"late chunking enabled! num chunks: {len(chunks)}")

    trackers.actionTracker.track_think("late_chunk", LANGUAGE_CODE, {"url": url})

    try:
        if not query.strip():
            raise ValueError("Empty question, returning full context")

        # 2. 获取 embeddings
        chunk_emb = get_embeddings(
            chunks,
            trackers.tokenTracker, {
                'task': "retrieval.passage",
                'dimensions': 1024,
                'late_chunking': True,
                'embedding_type': "float"
            }
        )["embeddings"]

        q_emb = get_embeddings(
            [query],
            trackers.token_tracker,
            {
                'task': "retrieval.query",
                'dimensions': 1024,
                'embedding_type': "float"
            }
        )["embeddings"][0]

        if len(chunk_emb) != len(chunks):
            log_error(f"Got {len(chunk_emb)} embeddings for {len(chunks)} chunks")

        # 3. 计算相似度
        sims = [cosine_similarity(q_emb, ce) for ce in chunk_emb]

        # 4. 滑窗选 snippet
        chunks_per_snippet = max(1, math.ceil(snippet_len / chunk_size))
        sims_copy = np.array(sims, dtype=float)

        snippets: List[str] = []
        for _ in range(num_snippets):
            best_start, best_score = 0, -math.inf
            for j in range(len(sims_copy) - chunks_per_snippet + 1):
                win_score = np.mean(sims_copy[j: j + chunks_per_snippet])
                if win_score > best_score:
                    best_score, best_start = win_score, j

            start_char = best_start * chunk_size
            end_char = min(start_char + snippet_len, len(long_context))
            snippets.append(long_context[start_char:end_char])

            # 标记已用
            sims_copy[best_start: best_start + chunks_per_snippet] = -math.inf

        # 5. 加标签返回
        tagged = "\n\n".join(
            f"<snippet-{idx + 1}>\n\n{snip}\n\n</snippet-{idx + 1}>"
            for idx, snip in enumerate(snippets)
        )
        return tagged

    except Exception as e:
        log_error("Error in late chunking: ", e)
        return long_context[: snippet_len * num_snippets]

# ------------------ 示例 ------------------
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     long = open("big_article.txt").read()
#     question = "What causes climate change?"
#     print(cherry_pick(question, long))
