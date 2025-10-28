# jina_reader.py
import os
import logging
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry

from utils.token_tracker import TokenTracker
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
TIMEOUT = 60
BASE_URL = "https://r.jina.ai/"


# 统一异常
class ReadUrlError(Exception):
    pass


def _build_session() -> requests.Session:
    """带重试的 session"""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    return sess


async def read_url(
        url: str,
        with_all_links: bool = False,
        tracker: Optional[TokenTracker] = None,
        with_all_images: bool = False,
) -> Dict[str, Any]:
    """
    等价于 TypeScript 的 readUrl
    返回: {"response": <ReadResponse dict>}
    """
    url = url.strip()
    if not url:
        raise ReadUrlError("URL cannot be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ReadUrlError("Invalid URL, only http and https URLs are supported")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "X-Md-Link-Style": "discarded",
    }
    if with_all_links:
        headers["X-With-Links-Summary"] = "all"
    if with_all_images:
        headers["X-With-Images-Summary"] = "true"
    else:
        headers["X-Retain-Images"] = "none"

    session = _build_session()
    try:
        resp = session.post(
            BASE_URL,
            json={"url": url},
            headers=headers,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ReadUrlError(f"Network error: {e}") from e
    except ValueError as e:
        raise ReadUrlError("Invalid JSON response") from e

    if not data.get("data"):
        raise ReadUrlError("Invalid response data")

    # 日志
    title = data["data"].get("title", "")
    logging.debug(f"Read: {title} ({url})")

    # token 统计
    tokens = data.get("data", {}).get("usage", {}).get("tokens") or 0
    token_tracker = tracker or TokenTracker()
    token_tracker.track_usage(
        "read",
        {
            'totalTokens': tokens,
            'promptTokens': len(url),
            'completionTokens': 0
        }
    )

    return {"response": data}
