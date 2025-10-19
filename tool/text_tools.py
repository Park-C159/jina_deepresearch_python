import json
import re
from pathlib import Path
from typing import Dict

from utils.get_log import get_logger

I18N_FILENAME = "i18n.json"  # 当前目录下的 i18nJSON 文件


def remove_html_tags(text: str) -> str:
    """移除字符串中的 HTML 标签。"""
    return re.sub(r'<[^>]*>', '', text, flags=re.MULTILINE | re.IGNORECASE)


def get_i18n_text(key: str, lang: str = 'en', params: Dict[str, str] | None = None):
    """
    等价于 TS 的 getI18nText：
    1) 优先使用指定语言；不存在则回退到 'en'
    2) 目标语言中没有该 key，则回退到 'en' 的该 key
    3) 若 'en' 也没有，则返回 key 本身
    4) 支持模板变量替换：${var}

    :param key: i18n 文本键名
    :param lang: 语言代码（默认 'en'）
    :param params: 模板变量字典，用于替换 ${var}
    """
    log = get_logger("i18n")
    # 占位：直接返回；你可接入实际的 i18n 实现
    path = Path.cwd() / "tool" / I18N_FILENAME

    if not path.exists():
        log.error(f"i18n file '{I18N_FILENAME}' not found in current directory.")
        i18nJSON = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as f:
                i18nJSON = json.load(f)
        except Exception as e:
            log.error(f"Failed to load '{I18N_FILENAME}': {e}")
            i18nJSON = {}

    i18n_data = i18nJSON
    # 语言回退
    if lang not in i18n_data:
        log.error(f"Language '{lang}' not found, falling back to English.")
        lang = "en"

    # 取对应语言的文本
    text = None
    lang_dict = i18n_data.get(lang, {})
    if isinstance(lang_dict, dict):
        text = lang_dict.get(key)

    # 键回退到英文
    if not text:
        log.error(f"Key '{key}' not found for language '{lang}', falling back to English.")
        text = i18n_data.get("en", {}).get(key)
        if not text:
            log.error(f"Key '{key}' not found for English either.")
            return key

    # 模板变量替换：${name}
    if params:
        for k, v in params.items():
            text = text.replace(f"${{{k}}}", v)

    return text
