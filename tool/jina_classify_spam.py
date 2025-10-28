# jina_classify.py
import os
import time
import logging
from typing import Optional, Callable

import requests
from dotenv import load_dotenv

load_dotenv()  # 把 .env 中的 JINA_API_KEY 加载到环境变量

JINA_API_URL = "https://api.jina.ai/v1/classify"
DEFAULT_CLASSIFIER_ID = "4a27dea0-381e-407c-bc67-250de45763dd"
DEFAULT_TIMEOUT = 5  # 秒

# 日志配置（可按需调整）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("jina_classify")


def _log_error(msg: str, **kwargs):
    """统一错误日志入口，方便后续接入 ELK/Sentry 等"""
    logger.error(msg, extra=kwargs)


async def classify_text(
        text: str,
        classifier_id: str = DEFAULT_CLASSIFIER_ID,
        timeout: float = DEFAULT_TIMEOUT,
        token_tracker: Optional[Callable[[str, int], None]] = None,
) -> bool:
    """
    调用 Jina Classify 接口，返回预测布尔值。
    超时或任何异常均返回 False，并记录日志。

    :param text: 待分类文本
    :param classifier_id: 分类器 ID
    :param timeout: 超时时间（秒）
    :param token_tracker: 回调函数，签名 (endpoint: str, total_tokens: int)
    :return: 预测结果，true/false 映射为 Python 布尔值
    """
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY is not set")

    payload = {
        "classifier_id": classifier_id,
        "input": [text],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        start = time.time()
        resp = requests.post(
            JINA_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        cost = time.time() - start
        logger.debug("Jina classify cost %.2fs", cost)

        resp.raise_for_status()
        data = resp.json()

        # token 用量追踪
        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        if token_tracker:
            token_tracker("classify", total_tokens)

        # 提取预测结果
        preds = data.get("data", [])
        if preds:
            return preds[0].get("prediction") == "true"
        return False

    except requests.exceptions.Timeout:
        _log_error("Classification request timed out", timeout=timeout)
        return False
    except Exception as e:
        _log_error("Error in classifying text", error=str(e))
        return False
