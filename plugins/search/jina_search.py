"""Jina AI 搜索插件"""
import os
from typing import Any, Dict, Optional

import aiohttp

from core.plugin_protocols import BaseSearchPlugin
from core.plugin_registry import register_plugin
from utils.token_tracker import TokenTracker


JINA_API_KEY = os.getenv("JINA_API_KEY")


@register_plugin("search", "jina")
class JinaSearchPlugin(BaseSearchPlugin):
    """Jina AI 搜索插件。支持通用搜索和 arxiv 域搜索。"""

    def initialize(self, cfg: Dict[str, Any]) -> None:
        self.api_key = cfg.get("api_key") or os.getenv("JINA_API_KEY")
        self.base_url = cfg.get("base_url") or "https://svip.jina.ai/"
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY not found")

    def health_check(self) -> bool:
        return bool(self.api_key)

    async def search(
        self,
        query: Dict[str, Any],
        domain: Optional[str] = None,
        num: Optional[int] = None,
        meta: Optional[str] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if domain != "arxiv":
            domain = None

        payload = dict(query)
        payload.update({"domain": domain, "num": num, "meta": meta})

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if not data.get("results") or not isinstance(data["results"], list):
            raise ValueError("Invalid response format from Jina")

        # token 追踪
        token_tracker = tracker or TokenTracker()
        prompt_length = len(query.get("q", ""))
        credits = data.get("meta", {}).get("credits", 0)
        token_tracker.track_usage("search", {
            "totalTokens": credits,
            "promptTokens": prompt_length,
            "completionTokens": 0,
        })

        return {"response": data}
