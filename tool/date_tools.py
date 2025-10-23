from datetime import datetime, timedelta
from typing import Optional


def format_date_range(tbs: str) -> str:
    """
    将 Google SERP 的 tbs 参数转成人类可读的时间段描述。
    """
    now = datetime.now()
    search_dt: Optional[datetime] = None
    fmt = "full"  # 默认完整格式

    match tbs:
        case "qdr:h":
            search_dt = now - timedelta(hours=1)
            fmt = "hour"
        case "qdr:d":
            search_dt = now - timedelta(days=1)
            fmt = "day"
        case "qdr:w":
            search_dt = now - timedelta(weeks=1)
            fmt = "day"
        case "qdr:m":
            search_dt = now - timedelta(days=30)
            fmt = "day"
        case "qdr:y":
            search_dt = now - timedelta(days=365)
            fmt = "year"
        case _:
            search_dt = None

    if search_dt is None:
        return ""

    start = format_date_based_on_type(search_dt, fmt)
    end = format_date_based_on_type(now, fmt)
    return f"Between {start} and {end}"


def format_date_based_on_type(dt: datetime, fmt: str) -> str:
    """按类型输出精简格式"""
    match fmt:
        case "hour":
            return dt.strftime("%H:%M")
        case "day":
            return dt.strftime("%Y-%m-%d")
        case "year":
            return dt.strftime("%Y")
        case _:  # full
            return dt.strftime("%Y-%m-%d %H:%M:%S")
