"""
Brave Search 插件 —— 示例：如何接入一个全新的搜索工具。

步骤回顾：
1. 在 plugins/search/ 下新建本文件
2. 继承 BaseSearchPlugin，实现 initialize / health_check / search
3. 使用 @register_plugin("search", "brave") 注册
4. 设置环境变量 SEARCH_PROVIDER=brave
5. 重启服务
"""
import os
from typing import Any, Dict, Optional

import aiohttp

from core.plugin_protocols import BaseSearchPlugin
from core.plugin_registry import register_plugin
from utils.token_tracker import TokenTracker


@register_plugin("search", "brave")
class BraveSearchPlugin(BaseSearchPlugin):
    """
    Brave Search API 插件。
    文档: https://api.search.brave.com/
    """

    def initialize(self, cfg: Dict[str, Any]) -> None:
        self.api_key = cfg.get("api_key") or os.getenv("BRAVE_API_KEY")
        self.base_url = cfg.get("base_url") or "https://api.search.brave.com/res/v1/web/search"
        if not self.api_key:
            raise RuntimeError("BRAVE_API_KEY not found. Get one at https://api.search.brave.com/")

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
        q = query.get("q", "")
        if not q:
            raise ValueError("Missing query text (expected query['q'])")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": q,
            "count": num or 20,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        # 适配为统一格式：Brave 返回的是 {"web": {"results": [...]}}
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
                "date": item.get("age"),
            })

        token_tracker = tracker or TokenTracker()
        token_tracker.track_usage("search", {
            "totalTokens": 0,
            "promptTokens": len(q),
            "completionTokens": 0,
        })

        return {"response": {"results": results, "meta": data.get("query", {})}}
