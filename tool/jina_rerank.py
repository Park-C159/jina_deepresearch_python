import asyncio
import os
import aiohttp
from typing import List, Dict, Any, Optional
from utils.get_log import get_logger

from dotenv import load_dotenv

from utils.token_tracker import TokenTracker

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_URL = 'https://api.jina.ai/v1/rerank'

MAX_RPS = 5
_sem = asyncio.Semaphore(MAX_RPS)

async def rerank_documents(
        query: str,
        documents,
        tracker=None,
        batch_size=2000,
):
    """
    Python 版 Jina rerank 接口封装，批次并发调用。
    返回格式: { "results": [ { "index": int, "relevance_score": float, "document": {"text": str} }, ... ] }
    """
    log = get_logger("rerank_documents")
    try:
        if not JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY is not set")

        # 拆批次
        batches = [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]
        log.debug(f"Reranking {len(documents)} documents in {len(batches)} batches")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        }

        async def post_batch(batch: List[str], batch_idx: int) -> List[Dict[str, Any]]:
            start_idx = batch_idx * batch_size
            payload = {
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "top_n": len(batch),
                "documents": batch,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(JINA_API_URL, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            # 记录 token
            total_tokens = data.get("usage", {}).get("total_tokens", 0)
            (tracker or TokenTracker()).track_usage(
                "rerank", {"promptTokens": total_tokens, "completionTokens": 0, "totalTokens": total_tokens}
            )

            # 把批次内 index 映射回全局 index
            return [
                {
                    **res,
                    "originalIndex": start_idx + res["index"],
                }
                for res in data["results"]
            ]

        # 并发执行所有批次
        batch_results = await asyncio.gather(*(post_batch(b, i) for i, b in enumerate(batches)))

        # 拍平 + 按 relevance_score 降序
        all_results = [res for batch in batch_results for res in batch]
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # 整理成最终格式
        final_results = [
            {"index": r["originalIndex"], "relevance_score": r["relevance_score"], "document": r["document"]}
            for r in all_results
        ]
        return {"results": final_results}
    except Exception as e:
        log.error(f'Reranking error: {e}')
        return {
            "results": []
        }




