import re


def remove_html_tags(text: str) -> str:
    """移除字符串中的 HTML 标签。"""
    return re.sub(r'<[^>]*>', '', text, flags=re.MULTILINE | re.IGNORECASE)
