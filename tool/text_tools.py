import json
import logging
import random
import re
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from utils.get_log import get_logger
from bs4 import BeautifulSoup

I18N_FILENAME = "i18n.json"  # 当前目录下的 i18nJSON 文件

def escape_regexp(string: str) -> str:
    """
    转义正则表达式特殊字符。
    等价于 JavaScript 的 escapeRegExp():
    将正则中所有特殊字符（如 . * + ? ^ $ { } ( ) | [ ] \）前加上反斜杠。
    """
    return re.sub(r'([.*+?^${}()|\[\]\\])', r'\\\1', string)


def count_char(text: str, char: str) -> int:
    """
    统计字符串 text 中某个字符 char 出现的次数。
    等价于 JavaScript 的 countChar():
      (text.match(new RegExp(escapeRegExp(char), 'g')) || []).length
    """
    pattern = re.compile(escape_regexp(char))
    return len(pattern.findall(text))
def process_formatted_text(text, open_marker, close_marker):
    """
    处理带有 markdown 格式标记的文本，
    将冒号 (:) 或全角冒号 (：) 移到标记外部。
    """
    try:
        pattern = re.compile(
            rf"{escape_regexp(open_marker)}(.*?){escape_regexp(close_marker)}", re.DOTALL
        )

        def replacer(match):
            content = match.group(1)
            if ":" in content or "：" in content:
                # 统计冒号数量
                standard_colons = count_char(content, ":")
                wide_colons = count_char(content, "：")

                # 移除冒号并清理空格
                trimmed = re.sub(r"[:：]", "", content).strip()

                # 在格式标记后追加原数量的冒号
                return f"{open_marker}{trimmed}{close_marker}{':' * standard_colons}{'：' * wide_colons}"
            return match.group(0)

        return pattern.sub(replacer, text)

    except Exception:
        return text


def repair_markdown_final(markdown):
    """
    修复 Markdown 文本的格式问题。
    包括：
    1. 删除非法字符与 <center> 标签
    2. 移除表格外的 <hr> 与 <br>
    3. 修复不匹配的 * / ** / *** 等格式标记
    """
    try:
        repaired = markdown or ""

        # === 1️⃣ 清理非法字符与 <center> 标签 ===
        repaired = repaired.replace("�", "")
        repaired = re.sub(r"</?center>", "", repaired)

        # === 2️⃣ 收集表格区域 (HTML 或 Markdown) ===
        table_regions = []

        # HTML 表格
        for m in re.finditer(r"<table[\s\S]*?</table>", repaired):
            table_regions.append((m.start(), m.end()))

        # Markdown 表格
        lines = repaired.split("\n")
        in_md_table = False
        md_table_start = 0
        for i, line in enumerate(lines):
            trimmed = line.strip()
            if trimmed.startswith("|") and "|" in trimmed[1:]:
                if not in_md_table:
                    in_md_table = True
                    md_table_start = repaired.find(lines[i])
            elif in_md_table and trimmed == "":
                in_md_table = False
                end_idx = repaired.find(lines[i - 1]) + len(lines[i - 1])
                table_regions.append((md_table_start, end_idx))

        if in_md_table:
            table_regions.append((md_table_start, len(repaired)))

        def is_in_table(idx):
            return any(start <= idx < end for start, end in table_regions)

        # === 3️⃣ 删除表格外的 <hr> 和 <br> ===
        result_chars = []
        i = 0
        while i < len(repaired):
            if repaired.startswith("<hr>", i) and not is_in_table(i):
                i += 4
            elif repaired.startswith("<br>", i) and not is_in_table(i):
                i += 4
            else:
                result_chars.append(repaired[i])
                i += 1

        repaired = "".join(result_chars)

        # === 4️⃣ 修复星号格式不匹配 ===
        formatting_patterns = [
            ("****", "****"),  # 四个星号
            ("****", "***"),  # 4 开 3 闭
            ("***", "****"),  # 3 开 4 闭
            ("***", "***"),  # 3 个星号
            ("**", "**"),  # 粗体
            ("*", "*"),  # 斜体
        ]

        for open_token, close_token in formatting_patterns:
            repaired = process_formatted_text(repaired, open_token, close_token)

        return repaired

    except Exception:
        # 出错则返回原文
        return markdown


def sanitize_cell(content):
    """
    清理 Markdown 表格单元格内容：
    1. 去除多余空格
    2. 转义管道符 |
    3. 保留换行（替换为 <br>）
    4. 保留已有 <br> 标签
    5. 修复被转义的 markdown 粗体、斜体等符号
    """
    if content is None:
        return ""

    # 1️⃣ 去除首尾空格
    sanitized = content.strip()

    # 2️⃣ 转义管道符号 |
    sanitized = sanitized.replace("|", "\\|")

    # 3️⃣ 将换行符替换为 <br>
    sanitized = sanitized.replace("\n", "<br>")

    # 4️⃣ 恢复 HTML 实体形式的 <br>
    sanitized = sanitized.replace("&lt;br&gt;", "<br>")

    # 5️⃣ 修复转义过的 Markdown 标记
    sanitized = (
        sanitized
        .replace("\\*\\*", "**")  # 修复粗体 **
        .replace("\\*", "*")  # 修复列表符 *
        .replace("\\_", "_")  # 修复斜体 _
    )

    return sanitized


def convertSingleHtmlTableToMd(html_table):
    """
    将单个 HTML 表格 (<table>...</table>) 转换为 Markdown 表格。
    保留粗体、斜体与换行符。
    """
    try:
        soup = BeautifulSoup(html_table, "html.parser")
        table = soup.find("table")
        if not table:
            return None

        # === 1️⃣ 提取表头 ===
        headers = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th"):
                headers.append(sanitize_cell(th.get_text()))
        else:
            # 如果没有 thead，则尝试从 tbody 第一行提取
            first_row = table.find("tr")
            if first_row:
                headers = [sanitize_cell(cell.get_text()) for cell in first_row.find_all(["th", "td"])]

        if not headers:
            return None  # 没有表头，无法生成有效 Markdown 表格

        # === 2️⃣ 构造 Markdown 表格头 ===
        md_table = "| " + " | ".join(headers) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # === 3️⃣ 遍历表格行 ===
        rows = table.find_all("tr")

        for idx, row in enumerate(rows):
            # 若没有 thead 且当前行是第一行，则跳过表头
            if not thead and idx == 0:
                continue

            cells = []
            for td in row.find_all("td"):
                content = td.decode_contents().strip()

                # 支持 Markdown 格式：转换 HTML 粗体/斜体标签
                content = (
                    content.replace("<strong>", "**").replace("</strong>", "**")
                    .replace("<b>", "**").replace("</b>", "**")
                    .replace("<em>", "_").replace("</em>", "_")
                    .replace("<i>", "_").replace("</i>", "_")
                    .replace("<br>", "<br>").replace("<br/>", "<br>")
                    .replace("</p>", "<br>").replace("<p>", "")
                )

                # 处理 HTML 列表（转为项目符号）
                if td.find_all("li"):
                    list_items = [li.get_text(strip=True) for li in td.find_all("li")]
                    content = "<br>".join(f"• {item}" for item in list_items)

                # 去除残余标签，但保留 <br>
                content = re.sub(r"<(?!/?br\b)[^>]+>", "", content)
                cells.append(sanitize_cell(content))

            # 确保行列数一致
            while len(cells) < len(headers):
                cells.append("")

            md_table += "| " + " | ".join(cells) + " |\n"

        return md_table.strip()

    except Exception as error:
        logging.error("Error converting single HTML table:", {"error": str(error)})
        return None


# 假设 repairMarkdownFootnotes 已在同文件或其他模块定义
# from .repair_markdown_footnotes import repairMarkdownFootnotes

def fixCodeBlockIndentation(markdown_text):
    """
    修复 Markdown 中文本块 (```code```) 的缩进。
    - 维持列表内代码块的正确缩进
    - 保留代码块内部原有缩进
    - 修复开关 fence 对齐不一致的问题
    """

    lines = markdown_text.split("\n")
    result = []
    code_block_stack = []  # Stack of dicts: {indent, language, listIndent}

    for i, line in enumerate(lines):
        # 判断是否是代码块 fence 行
        if line.lstrip().startswith("```"):
            indent = line[: line.find("```")] if "```" in line else ""
            rest_of_line = line.lstrip()[3:].strip()

            if not code_block_stack:
                # ======= 开始代码块 =======
                list_indent = ""
                if i > 0:
                    # 向上找 3 行内是否有列表符号（*, -, +, 1. 等）
                    for j in range(i - 1, max(-1, i - 4), -1):
                        prev_line = lines[j]
                        if re.match(r"^\s*(?:[*\-+]|\d+\.)\s", prev_line):
                            m = re.match(r"^(\s*)", prev_line)
                            if m:
                                list_indent = m.group(1)
                            break
                code_block_stack.append({
                    "indent": indent,
                    "language": rest_of_line,
                    "listIndent": list_indent
                })
                result.append(line)
            else:
                # ======= 结束代码块 =======
                opening = code_block_stack.pop() if code_block_stack else None
                if opening:
                    result.append(f"{opening['indent']}```")
                else:
                    result.append(line)

        elif code_block_stack:
            # ======= 在代码块内部 =======
            opening = code_block_stack[-1]

            if line.strip():
                # 计算基础缩进
                base_indent = (
                    opening["listIndent"] + "    "
                    if opening["listIndent"]
                    else opening["indent"]
                )

                # 当前行缩进
                m = re.match(r"^(\s*)", line)
                line_indent = m.group(1) if m else ""

                # 找公共前缀（markdown结构缩进）
                common_prefix = ""
                for a, b in zip(line_indent, opening["indent"]):
                    if a == b:
                        common_prefix += a
                    else:
                        break

                content_after_common = line[len(common_prefix):]
                result.append(f"{base_indent}{content_after_common}")
            else:
                # 空行原样保留
                result.append(line)
        else:
            # ======= 非代码块行 =======
            result.append(line)

    return "\n".join(result)


def convertHtmlTablesToMd(md_string):
    """
    将 Markdown 文本中嵌入的 HTML 表格 (<table>...</table>)
    转换为 Markdown 表格格式（或自定义格式）。
    """
    try:
        result = md_string

        # 仅当存在 <table> 时才处理
        if "<table" in md_string:
            # 匹配任意 HTML 表格（包括带属性的）
            table_regex = re.compile(r"<table(?:\s+[^>]*)?>([\s\S]*?)</table>", re.MULTILINE)

            for match in table_regex.finditer(md_string):
                html_table = match.group(0)
                converted_table = convertSingleHtmlTableToMd(html_table)

                if converted_table:
                    result = result.replace(html_table, converted_table)

        return result

    except Exception as error:
        logging.error("Error converting HTML tables to Markdown:", {"error": str(error)})
        return md_string


def repairMarkdownFootnotesOuter(markdown_string):
    """
    清理 markdown 文本：
    1. 移除包裹在 ```markdown 或 ```html 的代码块；
    2. 提取脚注定义部分；
    3. 从脚注中提取引用 (url, title, exactQuote)；
    4. 调用 repairMarkdownFootnotes() 生成标准脚注；
    若没有引用则返回原文。
    """
    if not markdown_string:
        return ""

    # 去除首尾空白
    markdown_string = markdown_string.strip()

    # === 1️⃣ 移除 fenced code blocks ```markdown 或 ```html ===
    code_block_regex = re.compile(r"```(markdown|html)\n([\s\S]*?)\n```", re.MULTILINE)
    processed_string = markdown_string

    for match in code_block_regex.finditer(markdown_string):
        entire_match = match.group(0)
        code_content = match.group(2)
        processed_string = processed_string.replace(entire_match, code_content)

    markdown_string = processed_string

    # === 2️⃣ 提取脚注定义 ===
    footnote_def_regex = re.compile(r"\[\^(\d+)]:\s*(.*?)(?=\n\[\^|$)", re.DOTALL)
    references = []

    content_part = markdown_string
    footnotes_part = ""

    # 找到脚注定义的起始位置
    first_match = re.search(r"\[\^(\d+)]:", markdown_string)
    if first_match:
        start_index = first_match.start()
        content_part = markdown_string[:start_index]
        footnotes_part = markdown_string[start_index:]

    # === 3️⃣ 遍历脚注定义 ===
    for m in footnote_def_regex.finditer(footnotes_part):
        if not m.group(2):
            continue

        content = m.group(2).strip()

        # 匹配末尾 URL 链接格式：[title](url)
        url_match = re.search(r"\s*\[([^\]]+)]\(([^)]+)\)\s*$", content)
        url = ""
        title = ""

        if url_match:
            title = url_match.group(1)
            url = url_match.group(2)
            # 移除 URL 部分，保留引文内容
            content = re.sub(re.escape(url_match.group(0)), "", content).strip()

        # 仅当内容、标题、URL 都存在时记录引用
        if content and title and url:
            references.append({
                "exactQuote": content,
                "url": url,
                "title": title
            })

    # === 4️⃣ 如果找到有效引用，则调用 repairMarkdownFootnotes 进行处理 ===
    if references:
        return repairMarkdownFootnotes(content_part, references)

    # === 5️⃣ 否则返回原 markdown ===
    return markdown_string


def build_md_from_answer(answer):
    """
    构建 Markdown 格式答案
    """
    return repairMarkdownFootnotes(
        answer.get("answer") or answer.get("mdAnswer") or "",
        answer.get("references")
    )


def repairMarkdownFootnotes(markdown_string, references=None):
    """
    修复 Markdown 脚注与引用格式。
    """

    # 正则定义
    footnote_regex = re.compile(r"\[(\^(\d+)|(\d+)\^|(\d+))]")
    grouped_footnote_regex = re.compile(r"\[\^(\d+)(?:,\s*\^(\d+))+]")
    partial_grouped_footnote_regex = re.compile(r"\[\^(\d+)(?:,\s*(\d+))+]")

    # 引用格式化函数
    def format_references(refs):
        valid_refs = [
            ref for ref in refs
            if ref and ref.get("url") and ref.get("title") and ref.get("exactQuote")
        ]

        formatted = []
        for i, ref in enumerate(valid_refs):
            clean_quote = re.sub(r"[^\w\s]", " ", ref.get("exactQuote", ""))
            clean_quote = re.sub(r"\s+", " ", clean_quote).strip()

            citation = f"[^{i + 1}]: {clean_quote}"
            url = ref.get("url")
            if not url:
                formatted.append(citation)
                continue

            domain_name = urlparse(url).hostname or ""
            domain_name = domain_name.replace("www.", "")
            title = ref.get("title") or domain_name
            formatted.append(f"{citation} [{title}]({url})")

        return "\n\n".join(formatted)

    # case 1: 没有引用
    if not references or len(references) == 0:
        markdown_string = partial_grouped_footnote_regex.sub(
            lambda m: ", ".join([f"[^{num}]" for num in re.findall(r"\d+", m.group(0))]),
            markdown_string
        )
        markdown_string = grouped_footnote_regex.sub(
            lambda m: ", ".join([f"[^{num}]" for num in re.findall(r"\d+", m.group(0))]),
            markdown_string
        )
        return footnote_regex.sub("", markdown_string)

    # 统一格式： [1^] → [^1], [1] → [^1]
    processed = re.sub(r"\[(\d+)\^]", lambda m: f"[^{m.group(1)}]", markdown_string)
    processed = re.sub(r"\[(\d+)]", lambda m: f"[^{m.group(1)}]", processed)

    # 修复成分组脚注
    processed = grouped_footnote_regex.sub(
        lambda m: ", ".join([f"[^{num}]" for num in re.findall(r"\d+", m.group(0))]),
        processed
    )
    processed = partial_grouped_footnote_regex.sub(
        lambda m: ", ".join([f"[^{num}]" for num in re.findall(r"\d+", m.group(0))]),
        processed
    )

    # 提取所有脚注
    standard_regex = re.compile(r"\[\^(\d+)]")
    footnotes = [m.group(1) for m in standard_regex.finditer(processed)]

    # 删除无对应引用的脚注
    cleaned = processed
    for fn in footnotes:
        if int(fn) > len(references):
            cleaned = re.sub(rf"\[\^{fn}]", "", cleaned)

    # 获取清理后的脚注
    valid_footnotes = [m.group(1) for m in standard_regex.finditer(cleaned)]

    # case 2: 没有脚注但有引用 → 自动追加引用
    if not valid_footnotes:
        appended_citations = "".join([f"[^{i + 1}]" for i in range(len(references))])
        formatted_refs = format_references(references)
        return f"""
{cleaned}

⁜{appended_citations}

{formatted_refs}
""".strip()

    # 检查是否需要重新编号
    needs_correction = (
            (len(valid_footnotes) == len(references) and all(n == valid_footnotes[0] for n in valid_footnotes)) or
            (all(n == valid_footnotes[0] for n in valid_footnotes) and int(valid_footnotes[0]) > len(references)) or
            (len(valid_footnotes) > 0 and all(int(n) > len(references) for n in valid_footnotes))
    )

    # case 3: 引用多于脚注 → 自动补齐未使用的引用
    if len(references) > len(valid_footnotes) and not needs_correction:
        used = {int(n) for n in valid_footnotes}
        unused_refs = "".join([
            f"[^{i + 1}]" if (i + 1) not in used else ""
            for i in range(len(references))
        ])
        formatted_refs = format_references(references)
        return f"""
{cleaned}

⁜{unused_refs}

{formatted_refs}
""".strip()

    # case 4: 不需要修正，直接输出格式化引用
    if not needs_correction:
        return f"""
{cleaned}

{format_references(references)}
""".strip()

    # case 5: 需要重新编号
    current_index = 0

    def replace_fn(_):
        nonlocal current_index
        current_index += 1
        return f"[^{current_index}]"

    corrected = standard_regex.sub(replace_fn, cleaned)

    return f"""
{corrected}

{format_references(references)}
""".strip()


def remove_html_tags(text: str) -> str:
    """移除字符串中的 HTML 标签。"""
    return re.sub(r'<[^>]*>', '', text, flags=re.MULTILINE | re.IGNORECASE)


def choose_k(a: list[str], k: int) -> list[str]:
    temp = a.copy()
    random.shuffle(temp)
    return temp[:k]


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
