import os

import aiohttp
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from utils.get_log import get_logger
from utils.token_tracker import TokenTracker

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
from typing import Optional, Callable, Dict, List, Any
import logging


# 假定 LanguageModelUsage 类型如下:
class LanguageModelUsage:
    def __init__(self, promptTokens=0, completionTokens=0, totalTokens=0):
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens

    def __add__(self, other):
        return LanguageModelUsage(
            self.promptTokens + other.promptTokens,
            self.completionTokens + other.completionTokens,
            self.totalTokens + other.totalTokens,
        )

    def __repr__(self):
        return f"LanguageModelUsage(promptTokens={self.promptTokens}, completionTokens={self.completionTokens}, totalTokens={self.totalTokens})"


async def search(query: Dict[str, Any],
           domain: Optional[str] = None,
           num: Optional[int] = None,
           meta: Optional[str] = None,
           tracker: Optional['TokenTracker'] = None) -> Dict[str, Any]:
    """
    【向后兼容】直接代理到 plugins.search.jina_search 插件。
    新代码建议直接使用 core.plugin_manager.get_plugin_instance("search", "jina").search(...)
    """
    from core.plugin_manager import get_plugin_instance
    plugin = get_plugin_instance("search", "jina", config={})
    return await plugin.search(query, domain=domain, num=num, meta=meta, tracker=tracker)


async def milvus_search(
        query: Dict[str, Any],
        domain: Optional[str] = None,
        num: Optional[int] = None,
        meta: Optional[str] = None,
        tracker: Optional['TokenTracker'] = None
) -> Dict[str, Any]:
    """
    【向后兼容】直接代理到 plugins.search.milvus_search 插件。
    """
    from core.plugin_manager import get_plugin_instance
    plugin = get_plugin_instance("search", "milvus", config={})
    return await plugin.search(query, domain=domain, num=num, meta=meta, tracker=tracker)
