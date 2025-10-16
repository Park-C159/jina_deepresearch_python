import os

import requests
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from utils.get_log import get_logger

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


class TokenTracker:
    def __init__(self, budget: Optional[int] = None):
        self.usages: List[Dict[str, Any]] = []
        self.budget: Optional[int] = budget
        self._usage_callbacks: List[Callable[[LanguageModelUsage], None]] = []

    def on_usage(self, callback: Callable[[LanguageModelUsage], None]):
        self._usage_callbacks.append(callback)

    def track_usage(self, tool: str, usage: LanguageModelUsage):
        u = {'tool': tool, 'usage': usage}
        self.usages.append(u)
        # Emit 'usage' event
        for cb in self._usage_callbacks:
            cb(usage)

    def get_total_usage(self) -> LanguageModelUsage:
        acc = LanguageModelUsage()
        for item in self.usages:
            usage = item['usage']
            scaler = 1  # 按原 ts 逻辑
            acc.promptTokens += usage.promptTokens * scaler
            acc.completionTokens += usage.completionTokens * scaler
            acc.totalTokens += usage.totalTokens * scaler
        return acc

    def get_total_usage_snake_case(self) -> dict:
        acc = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        for item in self.usages:
            usage = item['usage']
            scaler = 1
            acc['prompt_tokens'] += usage.promptTokens * scaler
            acc['completion_tokens'] += usage.completionTokens * scaler
            acc['total_tokens'] += usage.totalTokens * scaler
        return acc

    def get_usage_breakdown(self) -> Dict[str, int]:
        acc: Dict[str, int] = {}
        for item in self.usages:
            tool = item['tool']
            usage = item['usage']
            acc[tool] = acc.get(tool, 0) + usage.totalTokens
        return acc

    def print_summary(self):
        breakdown = self.get_usage_breakdown()
        total = self.get_total_usage()
        logging.info(f"Token Usage Summary: budget={self.budget}, total={total}, breakdown={breakdown}")

    def reset(self):
        self.usages = []


def search(query: Dict[str, Any],
           domain: Optional[str] = None,
           num: Optional[int] = None,
           meta: Optional[str] = None,
           tracker: Optional['TokenTracker'] = None) -> Dict[str, Any]:
    """
    搜索函数，将 TypeScript search 函数转为 Python 实现。
    :param query: 搜索参数
    :param domain: 指定搜索域（只有'arxiv'有效，否则为通用搜索）
    :param num: 返回结果数量
    :param meta: 其他元信息
    :param tracker: TokenTracker对象
    :return: 返回搜索结果的字典，键为'response'
    """
    log = get_logger("search")
    try:
        if domain != "arxiv":
            domain = None  # 默认为通用搜索

        payload = dict(query)  # 确保是可变dict
        payload.update({
            'domain': domain,
            'num': num,
            'meta': meta
        })
        url = "https://svip.jina.ai/"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}',
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data.get('results') or not isinstance(data['results'], list):
            raise ValueError('Invalid response format')

        log.debug(f"Search results metadata: {data.get('meta')}")

        token_tracker = tracker or TokenTracker()
        # 假定 query['q'] 是查询字符串
        prompt_length = len(query.get('q', ''))
        credits = data.get('meta', {}).get('credits', 0)
        token_tracker.track_usage('search', {
            'totalTokens': credits,
            'promptTokens': prompt_length,
            'completionTokens': 0
        })

        return {'response': data}

    except Exception as e:
        log.error(f"Search error: {e}")
        raise

