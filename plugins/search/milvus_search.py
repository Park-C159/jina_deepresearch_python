"""Milvus 向量数据库搜索插件"""
import os
from typing import Any, Dict, Optional

import aiohttp

from core.plugin_protocols import BaseSearchPlugin
from core.plugin_registry import register_plugin
from utils.token_tracker import TokenTracker


@register_plugin("search", "milvus")
class MilvusSearchPlugin(BaseSearchPlugin):
    """Milvus 向量数据库搜索插件。"""

    def initialize(self, cfg: Dict[str, Any]) -> None:
        self.base_url = cfg.get("base_url") or os.getenv("MILVUS_BASE_URL", "http://192.168.12.162:5445/milvus/rerank_query")
        self.api_key = cfg.get("api_key") or os.getenv("MILVUS_API_KEY")
        self.collection = cfg.get("collection") or os.getenv("MILVUS_COLLECTION", "all_info")
        self.limit = cfg.get("limit", 100)
        self.radius = cfg.get("radius", 0.5)

    def health_check(self) -> bool:
        return bool(self.base_url)

    async def search(
        self,
        query: Dict[str, Any],
        domain: Optional[str] = None,
        num: Optional[int] = None,
        meta: Optional[str] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        query_text = query.get("q", "")
        if not query_text:
            raise ValueError("Missing query text (expected query['q'])")

        payload = {
            "query": query_text,
            "limit": self.limit,
            "count": num or 20,
            "collection_name": self.collection,
            "radius": self.radius,
            "output_fields": ["id", "title", "content"],
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if not isinstance(data, dict) or "data" not in data:
            raise ValueError("Invalid response format from Milvus server")

        token_tracker = tracker or TokenTracker()
        token_tracker.track_usage("vector_search", {
            "totalTokens": 0,
            "promptTokens": len(query_text),
            "completionTokens": 0,
        })

        return {"response": data}
