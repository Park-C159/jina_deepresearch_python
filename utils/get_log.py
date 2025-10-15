import logging, sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """获取一个已配置好的 logger，同名 logger 全局唯一。"""
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # 避免重复 addHandler
        return logger

    logger.setLevel(logging.DEBUG)  # 全局最低级别

    # 1) 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    console.setFormatter(console_fmt)

    # 2) 文件（按大小切分）
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        LOG_DIR / f"{name}.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
