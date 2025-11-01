import re
from typing import List, Tuple, Literal, Optional, Union

ChunkType = Literal["newline", "punctuation", "characters", "regex"]


class ChunkOptions:
    def __init__(
        self,
        type: ChunkType = "newline",
        value: Optional[Union[str, int]] = None,
        minChunkLength: int = 80,
    ):
        self.type = type
        self.value = value
        self.minChunkLength = minChunkLength


def chunk_text(
    text: str, options: Optional[ChunkOptions] = None
) -> dict[str, Union[List[str], List[Tuple[int, int]]]]:
    """
    将文本按指定策略切分，并返回满足最小长度的块及其在原字符串中的起止位置。

    :param text: 原始文本
    :param options: 切分配置，不传时默认按 newline 切分，最小长度 80
    :return: {"chunks": List[str], "chunk_positions": List[Tuple[int, int]]}
    """
    if options is None:
        options = ChunkOptions()

    min_len = options.minChunkLength
    chunks: List[str] = []

    # 1. 按策略生成初步 chunks
    if options.type == "newline":
        chunks = [c for c in text.splitlines() if (c.strip())]

    elif options.type == "punctuation":
        # 保留分隔符（中英文句号/问号/感叹号）
        chunks = [c for c in re.split(r"(?<=[.!?。！？])", text) if (c.strip())]

    elif options.type == "characters":
        chunk_size = int(options.value) if options.value else 1000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    elif options.type == "regex":
        if not options.value or not isinstance(options.value, str):
            raise ValueError("Regex pattern is required for regex chunking")
        pattern = re.compile(options.value)
        chunks = [c for c in pattern.split(text) if (c.strip())]

    else:
        raise ValueError("Invalid chunking type")

    # 2. 过滤掉长度不足的块，并记录起止位置
    filtered_chunks: List[str] = []
    filtered_positions: List[Tuple[int, int]] = []
    current_pos = 0

    for chunk in chunks:
        start = text.find(chunk, current_pos)
        if start == -1:  # 理论上不会发生，保险起见
            continue
        end = start + len(chunk)
        if len(chunk) >= min_len:
            filtered_chunks.append(chunk)
            filtered_positions.append((start, end))
        current_pos = end

    return {"chunks": filtered_chunks, "chunk_positions": filtered_positions}

