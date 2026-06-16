"""
用户自定义数据加载器 (Data Loader)
==================================
支持用户上传自定义数据文件，将其解析为既适合 **LLM 结构化提取**、
又适合 **图表代码生成（pandas/matplotlib）** 直接使用的文本表示。

支持的格式：
- 表格类：``.csv`` / ``.tsv`` / ``.xlsx`` / ``.xls``
- 半结构化：``.json``
- 纯文本：``.txt`` / ``.md`` / ``.markdown``

对表格类数据，生成的文本包含三部分：
1. 数据概览（行列数、列名、各列类型）；
2. 描述性统计（``DataFrame.describe``，含数值与类别列）；
3. 完整数据的 CSV 文本（超出行数上限时截断，并给出说明）。

这样下游的分析 / 画图代码既能看到字段含义与统计特征，
又能直接从 CSV 文本里把数据还原成 DataFrame 进行精确计算。
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd


# 表格写入文本时的最大行数，避免超长文本撑爆 LLM 上下文
MAX_TABLE_ROWS = 500
# 预览（给前端展示）时的最大行数
PREVIEW_ROWS = 10
# 纯文本 / JSON 读入时的最大字符数
MAX_TEXT_CHARS = 20000

# 上传的表格数据会被「落地」为真实 CSV 文件存放在该目录下（相对当前工作目录）。
# 这样下游生成的分析 / 绘图代码可以用 pandas 直接读取完整、准确的数据，
# 而不是依赖把全部数据塞进 LLM 文本（既会截断丢数据，也会让模型回吐数据时超长出错）。
UPLOAD_DATA_DIR = "uploaded_data"


class DataLoadError(Exception):
    """数据文件加载失败时抛出。"""


def _read_bytes_with_retry(file_path: str, retries: int = 5, delay: float = 0.15) -> bytes:
    """一次性把文件读入内存字节。

    Gradio 在 Windows 上会把上传文件放入临时目录，文件句柄可能尚未释放，
    直接用 pandas/open 打开该路径有概率触发 ``[Errno 13] Permission denied``
    （句柄竞态/被占用）。这里改为「读字节到内存」并对权限/占用类错误做重试，
    后续解析都基于内存数据进行，彻底规避对临时文件路径的反复打开。
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except (PermissionError, OSError) as e:  # 含 [Errno 13] Permission denied
            last_err = e
            time.sleep(delay * (attempt + 1))
    raise DataLoadError(
        f"无法读取文件（可能被占用或权限不足）: {os.path.basename(file_path)}: {last_err}"
    )


def _read_tabular(file_path: str) -> pd.DataFrame:
    """根据扩展名把表格类文件读成 DataFrame（基于内存字节，避免临时文件占用）。"""
    ext = os.path.splitext(file_path)[1].lower()
    data = _read_bytes_with_retry(file_path)
    if ext in (".csv",):
        # 自动尝试常见编码，兼容中文 Excel 导出的 GBK
        for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(io.BytesIO(data))
    if ext in (".tsv",):
        for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), sep="\t", encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return pd.read_csv(io.BytesIO(data), sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise DataLoadError(f"不支持的表格文件类型: {ext}")


def _sanitize_filename(name: str) -> str:
    """把文件名转成适合做磁盘文件名的安全字符串。"""
    base = os.path.splitext(os.path.basename(name))[0]
    safe = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in base)
    return safe or "data"


def _save_df_csv(df: pd.DataFrame, source_name: str) -> Optional[str]:
    """把 DataFrame 落地保存为真实 CSV 文件，返回其相对路径（失败返回 None）。

    下游分析 / 绘图代码可据此用 ``pd.read_csv`` 读取完整、准确的数据。
    使用 ``utf-8-sig`` 编码，兼容含中文的场景。
    """
    try:
        os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
        csv_path = os.path.join(UPLOAD_DATA_DIR, f"{_sanitize_filename(source_name)}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path
    except Exception:
        return None


def _dataframe_to_text(df: pd.DataFrame, source_name: str, csv_path: Optional[str] = None) -> str:
    """把 DataFrame 转换成结构化的文本描述。

    :param csv_path: 该数据已落地保存的真实 CSV 路径。若提供，则在文本中明确告知下游
        代码可以用 ``pd.read_csv`` 直接读取完整数据（避免依赖文本内联数据导致截断 / 丢失）。
    """
    n_rows, n_cols = df.shape

    # 1. 列信息（名称 + 类型）
    col_lines = [f"  - {col} ({dtype})" for col, dtype in df.dtypes.items()]
    columns_block = "\n".join(col_lines)

    # 2. 描述性统计（数值列 + 类别列）
    try:
        desc = df.describe(include="all").transpose()
        describe_block = desc.to_markdown()
    except Exception:
        describe_block = "（无法生成描述性统计）"

    # 3. 数据读取说明 / 完整数据
    if csv_path:
        # 已落地为真实 CSV：优先让下游代码直接读取该文件，数据最完整准确。
        norm_path = csv_path.replace("\\", "/")
        data_block = f"""## 数据文件（请优先用代码读取此文件获取完整数据）
完整数据已保存为 CSV 文件，**请在分析 / 绘图代码中使用如下方式读取完整数据**：
```python
import pandas as pd
df = pd.read_csv(r"{norm_path}")
```
注意：请勿凭空编造数据，所有数值都应来自上述文件。下面仅给出前若干行用于了解数据结构。

### 数据样例（前 {min(PREVIEW_ROWS, n_rows)} 行）
```csv
{df.head(PREVIEW_ROWS).to_csv(index=False)}
```
"""
    else:
        truncated = n_rows > MAX_TABLE_ROWS
        body_df = df.head(MAX_TABLE_ROWS) if truncated else df
        csv_text = body_df.to_csv(index=False)
        trunc_note = (
            f"\n注意：原始数据共 {n_rows} 行，此处仅展示前 {MAX_TABLE_ROWS} 行。"
            if truncated else ""
        )
        data_block = f"""## 完整数据（CSV 格式）{trunc_note}
```csv
{csv_text}
```
"""

    return f"""# 用户上传数据集：{source_name}

## 数据概览
- 数据行数：{n_rows}
- 数据列数：{n_cols}
- 字段列表：
{columns_block}

## 描述性统计
{describe_block}

{data_block}"""


def _json_to_text(file_path: str, source_name: str) -> Dict[str, Any]:
    """把 JSON 文件转成文本；若是记录数组则尝试转 DataFrame 表格化并落地为 CSV。

    :return: ``{"text": ..., "csv_path": Optional[str]}``
    """
    raw = _read_bytes_with_retry(file_path)
    data = json.loads(raw.decode("utf-8-sig"))

    # 形如 [{...}, {...}] 的记录数组：可表格化，分析价值最高
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        try:
            df = pd.json_normalize(data)
            csv_path = _save_df_csv(df, source_name)
            return {"text": _dataframe_to_text(df, source_name, csv_path), "csv_path": csv_path}
        except Exception:
            pass

    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    if len(pretty) > MAX_TEXT_CHARS:
        pretty = pretty[:MAX_TEXT_CHARS] + "\n...（内容过长已截断）"
    text = f"""# 用户上传数据集：{source_name}（JSON）

```json
{pretty}
```
"""
    return {"text": text, "csv_path": None}


def _plaintext_to_text(file_path: str, source_name: str) -> str:
    """读纯文本 / Markdown 文件（基于内存字节，避免临时文件占用）。"""
    raw = _read_bytes_with_retry(file_path)
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        content = raw.decode("utf-8", errors="ignore")

    if len(content) > MAX_TEXT_CHARS:
        content = content[:MAX_TEXT_CHARS] + "\n...（内容过长已截断）"
    return f"""# 用户上传数据：{source_name}

{content}
"""


def load_uploaded_data(file_path: str) -> Dict[str, Any]:
    """加载单个用户上传的数据文件。

    :param file_path: 上传文件在本地的路径
    :return: ``{"text": 供分析使用的文本, "preview": 前端预览(markdown), "name": 文件名,
        "kind": 类型, "csv_path": 落地的真实 CSV 路径(仅表格类，否则 None)}``
    :raises DataLoadError: 文件不存在或解析失败
    """
    if not file_path or not os.path.exists(file_path):
        raise DataLoadError(f"文件不存在: {file_path}")

    source_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    csv_path: Optional[str] = None
    try:
        if ext in (".csv", ".tsv", ".xlsx", ".xls"):
            df = _read_tabular(file_path)
            csv_path = _save_df_csv(df, source_name)
            text = _dataframe_to_text(df, source_name, csv_path)
            preview = _build_preview(df, source_name)
            kind = "table"
        elif ext == ".json":
            res = _json_to_text(file_path, source_name)
            text = res["text"]
            csv_path = res.get("csv_path")
            preview = f"**{source_name}**（JSON）已加载。"
            kind = "json"
        elif ext in (".txt", ".md", ".markdown"):
            text = _plaintext_to_text(file_path, source_name)
            preview = f"**{source_name}**（文本）已加载。"
            kind = "text"
        else:
            raise DataLoadError(f"不支持的文件类型: {ext}")
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"解析文件 {source_name} 失败: {e}") from e

    return {"text": text, "preview": preview, "name": source_name, "kind": kind, "csv_path": csv_path}


def _build_preview(df: pd.DataFrame, source_name: str) -> str:
    """构建前端展示用的 Markdown 数据预览。"""
    n_rows, n_cols = df.shape
    try:
        head_md = df.head(PREVIEW_ROWS).to_markdown(index=False)
    except Exception:
        head_md = df.head(PREVIEW_ROWS).to_string(index=False)
    return f"""**{source_name}** 已加载：{n_rows} 行 × {n_cols} 列

预览（前 {min(PREVIEW_ROWS, n_rows)} 行）：

{head_md}
"""


def load_multiple(file_paths: List[str]) -> Dict[str, Any]:
    """加载多个数据文件，合并为统一的分析文本。

    :param file_paths: 上传文件路径列表
    :return: ``{"text": 合并文本, "preview": 合并预览, "names": [文件名...],
        "csv_paths": [落地的真实 CSV 路径...], "errors": [...]}``
    """
    texts: List[str] = []
    previews: List[str] = []
    names: List[str] = []
    csv_paths: List[str] = []
    errors: List[str] = []

    for fp in file_paths or []:
        try:
            loaded = load_uploaded_data(fp)
            texts.append(loaded["text"])
            previews.append(loaded["preview"])
            names.append(loaded["name"])
            if loaded.get("csv_path"):
                csv_paths.append(loaded["csv_path"])
        except DataLoadError as e:
            errors.append(str(e))

    combined_text = "\n\n---\n\n".join(texts)
    combined_preview = "\n\n".join(previews)
    if errors:
        combined_preview += "\n\n**加载失败：**\n" + "\n".join(f"- {e}" for e in errors)

    return {"text": combined_text, "preview": combined_preview, "names": names,
            "csv_paths": csv_paths, "errors": errors}
