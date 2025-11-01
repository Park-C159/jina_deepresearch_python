import logging
import math
import re
import urllib.parse
from datetime import datetime

import aiohttp
import asyncio

from typing import Dict, List, Any

from tool.date_tools import format_date_based_on_type
from tool.image_tools import process_image
from tool.jina_classify_spam import classify_text
from tool.jina_latechunk import cherry_pick
from tool.jina_rerank import rerank_documents
from tool.read_tools import read_url
from tool.segments import chunk_text
from tool.text_tools import get_i18n_text
from utils.get_log import get_logger
from urllib.parse import urlparse
import re
import urllib.parse

from utils.schemas import LANGUAGE_CODE


def extract_url_parts(url_str):
    """
    提取 URL 的主机名（去掉 www. 前缀）和路径。
    若解析失败则返回空字符串。
    """
    try:
        parsed = urlparse(url_str)
        hostname = parsed.hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = parsed.path or ""
        return {"hostname": hostname, "path": path}
    except Exception as e:
        logging.error(f"Error parsing URL: {url_str}", error=e)
        return {"hostname": "", "path": ""}


def normalize_host_name(host_str):
    """
    标准化主机名：
      - 优先解析 URL 获取主机部分；
      - 若解析失败，则尝试去掉 'www.' 前缀并小写化；
    """
    extract = extract_url_parts(host_str)
    host = extract["hostname"]

    if not host:
        return host_str[4:].lower() if host_str.startswith("www.") else host_str.lower()

    return host


def fix_bad_url_md_links(md_content, all_urls):
    """
    修复 Markdown 链接中的“坏链接文本”：
    如果链接文本与 URL 完全相同，则尝试替换为更易读的格式：
    - 若在 all_urls 中有标题信息，则使用 [title - hostname](url)
    - 否则使用 [hostname](url)
    """

    def replace_link(match):
        text, url = match.group(1), match.group(2)

        # 如果链接文本与 URL 完全一致，则尝试修复
        if text == url:
            url_info = all_urls.get(url)

            try:
                hostname = urllib.urlparse(url).hostname or url
            except Exception:
                return match.group(0)  # URL 无效时返回原始链接

            if url_info:
                title = url_info.get("title")
                if title:
                    return f"[{title} - {hostname}]({url})"
                else:
                    return f"[{hostname}]({url})"
            else:
                # 若无元数据，只显示域名
                return f"[{hostname}]({url})"

        # 否则保持原始链接
        return match.group(0)

    # 匹配 Markdown 链接: [text](url)
    md_link_regex = re.compile(r"\[([^\]]+)]\(([^)]+)\)")
    return md_link_regex.sub(replace_link, md_content)


def normalize_url(
        url_string: str,
        debug: bool = False,
        options: dict = None
) -> str | None:
    """
    Normalize a URL string according to similar rules as the TS version.
    Returns normalized URL or None if invalid.
    """
    if options is None:
        options = {
            'removeAnchors': True,
            'removeSessionIDs': True,
            'removeUTMParams': True,
            'removeTrackingParams': True,
            'removeXAnalytics': True
        }

    def log_debug(msg, obj):
        print('[DEBUG]', msg, obj)

    def log_warning(msg):
        print('[WARN]', msg)

    try:
        url_string = re.sub(r'\s+', '', url_string).strip()
        if not url_string:
            raise ValueError("Empty URL")
        if url_string.startswith('https://google.com/') or url_string.startswith(
                'https://www.google.com') or url_string.startswith('https://baidu.com/s?'):
            raise ValueError('Google/baidu search link')
        if 'example.com' in url_string:
            raise ValueError('Example URL')

        # x.com / twitter.com /analytics
        if options.get('removeXAnalytics', True):
            x_com_pattern = re.compile(
                r'^(https?://(?:www\.)?(x\.com|twitter\.com)/([^/]+)/status/(\d+))/analytics(/)?(\?.*)?(#.*)?$',
                re.IGNORECASE
            )
            match = x_com_pattern.match(url_string)
            if match:
                clean_url = match.group(1)
                if match.group(6): clean_url += match.group(6)
                if match.group(7): clean_url += match.group(7)
                url_string = clean_url

        # Parse URL
        try:
            url = urllib.parse.urlparse(url_string)
        except Exception:
            raise ValueError('URL parse error')

        if url.scheme not in ('http', 'https'):
            raise ValueError('Unsupported protocol')

        hostname = url.hostname.lower() if url.hostname else ''
        if hostname.startswith('www.'):
            hostname = hostname[4:]

        port = url.port
        # Remove default port
        netloc = hostname
        if port and not ((url.scheme == 'http' and port == 80) or (url.scheme == 'https' and port == 443)):
            netloc += f':{port}'

        # Path normalization
        def decode_seg(seg):
            try:
                return urllib.parse.unquote(seg)
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode path segment: {seg}', {'error': str(e)})
                return seg

        path = '/'.join([decode_seg(s) for s in url.path.split('/')])
        path = re.sub(r'/+', '/', path)
        path = re.sub(r'/+$', '', path)
        path = path if path else '/'

        # Query param normalization
        search_params = urllib.parse.parse_qsl(url.query, keep_blank_values=True)

        def param_filter(key):
            if key == '':
                return False
            # Session IDs
            if options.get('removeSessionIDs', True) and re.match(
                    r'^(s|session|sid|sessionid|phpsessid|jsessionid|aspsessionid|asp\.net_sessionid)$', key, re.I):
                return False
            # UTM params
            if options.get('removeUTMParams', True) and re.match(r'^utm_', key, re.I):
                return False
            # tracking params
            if options.get('removeTrackingParams', True) and re.match(
                    r'^(ref|referrer|fbclid|gclid|cid|mcid|source|medium|campaign|term|content|sc_rid|mc_[a-z]+)$', key,
                    re.I):
                return False
            return True

        sorted_params = []
        for key, value in search_params:
            if not param_filter(key):
                continue
            try:
                decoded_value = urllib.parse.unquote(value)
                if urllib.parse.quote(decoded_value) == value:
                    value = decoded_value
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode query param {key}={value}', {'error': str(e)})
            sorted_params.append((key, value))

        sorted_params = sorted(sorted_params, key=lambda kv: kv[0])
        query = urllib.parse.urlencode(sorted_params)

        # Fragment normalization
        fragment = url.fragment
        if options.get('removeAnchors', True):
            fragment = ''
        elif fragment in ('', '/', 'top'):
            fragment = ''
        else:
            try:
                decoded_frag = urllib.parse.unquote(fragment)
                if urllib.parse.quote(decoded_frag) == fragment:
                    fragment = decoded_frag
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode fragment: #{fragment}', {'error': str(e)})

        # Build final URL
        final_url = urllib.parse.urlunparse((
            url.scheme,
            netloc,
            path,
            '',  # params
            query,
            fragment
        ))

        # Remove trailing slash except for home "/"
        if path != '/' and path.endswith('/'):
            path = path[:-1]
            final_url = urllib.parse.urlunparse((
                url.scheme,
                netloc,
                path,
                '',  # params
                query,
                fragment
            ))

        # Final decode check
        try:
            decoded_url = urllib.parse.unquote(final_url)
            if urllib.parse.quote(decoded_url) == final_url:
                final_url = decoded_url
        except Exception as e:
            if debug:
                log_debug('Failed to decode final URL', {'error': str(e)})

        return final_url
    except Exception as e:
        log_warning(f'Invalid URL "{url_string}": {e}')
        return None


async def get_last_modified(url: str) -> str | None:
    """
    Fetch last modified date for a URL via external API.
    """
    api_url = f'https://api-beta-datetime.jina.ai?url={urllib.parse.quote(url)}'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=10) as resp:
                data = await resp.json()
                if data.get('bestGuess') and data.get('confidence', 0) >= 70:
                    return data['bestGuess']
        return None
    except Exception as e:
        print('[ERROR] Failed to fetch last modified date', e)
        return None


def smart_merge_strings(str1: str, str2: str) -> str:
    # If either string is empty, return the other
    if not str1:
        return str2
    if not str2:
        return str1

    # Check if one string is entirely contained within the other
    if str2 in str1:
        return str1
    if str1 in str2:
        return str2

    # Find the maximum possible overlap length
    max_overlap = min(len(str1), len(str2))
    best_overlap_length = 0

    # Check for overlaps starting from the largest possible
    for overlap_length in range(max_overlap, 0, -1):
        end_of_str1 = str1[-overlap_length:]
        start_of_str2 = str2[:overlap_length]
        if end_of_str1 == start_of_str2:
            best_overlap_length = overlap_length
            break

    # If found overlap, merge without repeating that part
    if best_overlap_length > 0:
        return str1 + str2[best_overlap_length:]
    else:
        return str1 + str2


def sort_select_urls(all_urls, max_urls=70):
    """
    等价于 TypeScript 的 sortSelectURLs。
    根据 finalScore 降序排序，并返回前 max_urls 个结果。
    """
    if not all_urls:
        return []

    merged_items = []
    for r in all_urls:
        merged = smart_merge_strings(r.get("title"), r.get("description"))
        merged_items.append({
            "url": r.get("url"),
            "score": r.get("finalScore", 0),
            "merged": merged
        })

    # 过滤掉空 merged 的项
    filtered = [item for item in merged_items if item["merged"]]

    # 按 score 降序排序
    sorted_items = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)

    # 取前 max_urls 个
    return sorted_items[:max_urls]


def add_to_all_urls(r, all_urls, weight_delta=1):
    """
    r: SearchSnippet对象（dict或自定义类）
    all_urls: dict，key=标准化后的url，value=SearchSnippet对象
    weight_delta: 权重增量
    返回值: 如果新加入URL返回1，否则返回0
    """
    n_url = normalize_url(r['url'])
    if not n_url:
        return 0
    if n_url not in all_urls:
        all_urls[n_url] = r.copy()  # 确保新的对象
        all_urls[n_url]['weight'] = weight_delta
        return 1
    else:
        all_urls[n_url]['weight'] += weight_delta
        cur_desc = all_urls[n_url]['description']
        # 合并旧描述和新描述
        all_urls[n_url]['description'] = smart_merge_strings(cur_desc, r['description'])
        return 0


def extract_urls_with_description(text: str, context_window_size: int = 50):
    """
    从任意文本中提取所有 HTTP/HTTPS 链接，并为每个链接截取前后 context_window_size 个字符作为描述。
    返回纯 Python 原生 list[dict]，字段仅包含 url / description / title（title 始终为空字符串）。
    不引入任何自定义数据类、类型提示或第三方库。
    """
    # 与 TypeScript 版本功能等价的正则
    url_pattern = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*'
    )

    matches = []
    for match in url_pattern.finditer(text):
        url = match.group(0)

        # 去掉末尾常见标点
        while url and url[-1] in '.,;:!?)':
            url = url[:-1]

        matches.append({
            'url': url,
            'index': match.start(),
            'length': len(url)
        })

    if not matches:
        return []

    results = []
    for i, item in enumerate(matches):
        url = item['url']
        idx = item['index']
        length = item['length']

        # 计算上下文边界
        start_pos = max(0, idx - context_window_size)
        end_pos = min(len(text), idx + length + context_window_size)

        # 避免与相邻 URL 重叠
        if i > 0:
            prev_end = matches[i - 1]['index'] + matches[i - 1]['length']
            start_pos = max(start_pos, prev_end)
        if i < len(matches) - 1:
            next_start = matches[i + 1]['index']
            end_pos = min(end_pos, next_start)

        before_text = text[start_pos:idx].strip()
        after_text = text[idx + length:end_pos].strip()

        if before_text and after_text:
            description = f"{before_text} ... {after_text}"
        elif before_text:
            description = before_text
        elif after_text:
            description = after_text
        else:
            description = "No context available"

        # 合并多余空白
        description = " ".join(description.split())

        results.append({
            "url": url,
            "description": description,
            "title": ""
        })

    return results


def normalize_count(cnt: int, total: int) -> float:
    return cnt / total if total else 0.0


def extract_url_parts(url_str: str) -> dict:
    """
    从 URL 中提取 hostname 与 path，与 TS 版本 extractUrlParts 等价。
    """
    try:
        parsed = urlparse(url_str)
        hostname = parsed.hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = parsed.path or ""
        return {"hostname": hostname, "path": path}
    except Exception as e:
        logging.error(f"Error parsing URL: {url_str}", exc_info=e)
        return {"hostname": "", "path": ""}


def count_url_parts(url_items: list) -> dict:
    """
    统计 URL 的 hostname 和 path 前缀频次。
    与 TypeScript countUrlParts 完全等价。
    """
    hostname_count = {}
    path_prefix_count = {}
    total_urls = 0

    for item in url_items:
        if not item or not item.get("url"):
            continue
        total_urls += 1
        parts = extract_url_parts(item["url"])
        hostname = parts["hostname"]
        path = parts["path"]

        hostname_count[hostname] = hostname_count.get(hostname, 0) + 1

        segments = [s for s in path.split("/") if s]
        for i in range(len(segments)):
            prefix = "/" + "/".join(segments[: i + 1])
            path_prefix_count[prefix] = path_prefix_count.get(prefix, 0) + 1

    return {
        "hostnameCount": hostname_count,
        "pathPrefixCount": path_prefix_count,
        "totalUrls": total_urls
    }


def filter_urls(
        all_urls: Dict[str, Dict[str, Any]],
        visited_urls: List[str],
        bad_hostnames: List[str],
        only_hostnames: List[str]
) -> List[Dict[str, Any]]:
    """
    过滤 URL 记录。
    返回满足条件的 SearchSnippet 列表。
    """

    def extract_hostname(url: str) -> str:
        # 简单提取 hostname，可按需换成 urllib.parse
        url = url.lstrip("https://").lstrip("http://")
        return url.split("/")[0] if "/" in url else url

    filtered = []
    for url, snippet in all_urls.items():
        if url in visited_urls:
            continue
        hostname = extract_hostname(url)
        if hostname in bad_hostnames:
            continue
        if only_hostnames and hostname not in only_hostnames:
            continue
        filtered.append(snippet)
    return filtered


async def rank_urls(url_items: list, options: dict = None, trackers=None) -> list:
    """
    完整等价于 TypeScript 版本的 rankURLs()。
    :param url_items: list of dict，包含 url/title/description/weight 等字段
    :param options: 可配置的 boosting 参数
    :param trackers: 可选，包含 tokenTracker
    :return: 带有 boost 权重的排序结果列表
    """
    if options is None:
        options = {}

    # === 默认参数 ===
    freq_factor = options.get("freqFactor", 0.5)
    hostname_boost_factor = options.get("hostnameBoostFactor", 0.5)
    path_boost_factor = options.get("pathBoostFactor", 0.4)
    decay_factor = options.get("decayFactor", 0.8)
    jina_rerank_factor = options.get("jinaRerankFactor", 0.8)
    min_boost = options.get("minBoost", 0)
    max_boost = options.get("maxBoost", 5)
    question = options.get("question", "")
    boost_hostnames = options.get("boostHostnames", [])

    # === 统计 URL 部分 ===
    counts = count_url_parts(url_items)
    hostname_count = counts["hostnameCount"]
    path_prefix_count = counts["pathPrefixCount"]
    total_urls = counts["totalUrls"]

    # === Jina rerank 逻辑 ===
    if question.strip():
        unique_content_map = {}

        # Step 1: 按合并内容去重
        for idx, item in enumerate(url_items):
            merged = smart_merge_strings(item.get("title", ""), item.get("description", ""))
            unique_content_map.setdefault(merged, []).append(idx)

        unique_contents = list(unique_content_map.keys())
        unique_indices_map = list(unique_content_map.values())
        logging.debug(f"unique URLs: {len(url_items)} -> {len(unique_contents)}")

        token_tracker = getattr(trackers, "tokenTracker", None) if trackers else None
        rerank_result = await rerank_documents(question, unique_contents, token_tracker)
        for res in rerank_result.get("results", []):
            idx = res["index"]
            score = res["relevance_score"]
            boost = score * jina_rerank_factor
            for orig_idx in unique_indices_map[idx]:
                url_items[orig_idx]["jinaRerankBoost"] = boost

    # === 计算各项 boost ===
    boosted_items = []
    for item in url_items:
        if not item or not item.get("url"):
            logging.error(f"Skipping invalid item: {item}")
            boosted_items.append(item)
            continue

        parts = extract_url_parts(item["url"])
        hostname = parts["hostname"]
        path = parts["path"]

        freq = item.get("weight", 0)

        # Hostname boost
        hostname_freq = normalize_count(hostname_count.get(hostname, 0), total_urls)
        hostname_boost = hostname_freq * hostname_boost_factor
        if hostname in boost_hostnames:
            hostname_boost += 2

        # Path boost
        path_boost = 0.0
        segments = [s for s in path.split("/") if s]
        for i in range(len(segments)):
            prefix = "/" + "/".join(segments[: i + 1])
            prefix_count = path_prefix_count.get(prefix, 0)
            prefix_freq = normalize_count(prefix_count, total_urls)
            decayed = prefix_freq * (decay_factor ** i) * path_boost_factor
            path_boost += decayed

        freq_boost = (freq / total_urls * freq_factor) if total_urls else 0.0
        jina_rerank_boost = item.get("jinaRerankBoost", 0.0)

        final_score = hostname_boost + path_boost + freq_boost + jina_rerank_boost
        final_score = max(min(final_score, max_boost), min_boost)

        boosted_item = {
            **item,
            "freqBoost": freq_boost,
            "hostnameBoost": hostname_boost,
            "pathBoost": path_boost,
            "jinaRerankBoost": jina_rerank_boost,
            "finalScore": final_score,
        }
        boosted_items.append(boosted_item)

    boosted_items.sort(key=lambda x: x.get("finalScore", 0), reverse=True)
    return boosted_items


def keep_k_per_hostname(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    每个 hostname 最多保留 k 条记录，顺序不变。
    """

    def extract_hostname(url: str) -> str:
        url = url.lstrip("https://").lstrip("http://")
        return url.split("/")[0] if "/" in url else url

    # 快速判断是否需要筛选
    unique_host = {extract_hostname(r["url"]) for r in results if r.get("url")}
    if len(unique_host) <= 1:
        return results

    hostname_map: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []

    for item in results:
        host = extract_hostname(item["url"])
        cnt = hostname_map.get(host, 0)
        if cnt < k:
            filtered.append(item)
            hostname_map[host] = cnt + 1

    return filtered


async def process_urls(
        urls,
        context,
        all_knowledge,
        all_urls,
        visited_urls,
        bad_urls,
        image_objects,
        question,
        web_contents,
        with_images=False
):
    if not urls:
        return {"urlResults": [], "success": False}

    bad_hostnames: List[str] = []

    # ---------- 1. 记录阅读动作 ----------
    this_step = {
        "action": "visit",
        "think": get_i18n_text(
            "read_for",
            LANGUAGE_CODE,
            {"urls": ", ".join(urls)}
        ),
        "URL_target": urls,
    }
    context.actionTracker.track_action({"thisStep": this_step})

    # ---------- 2. 并行处理每个 URL ----------
    async def _process_single(url: str):
        nonlocal bad_hostnames
        try:
            normalized = normalize_url(url)
            if not normalized:
                return None
            url = normalized  # 统一用归一化后的 URL

            response = await read_url(url, True, context.tokenTracker, with_images)

            data = response["response"]["data"]
            guessed_time = await get_last_modified(url)
            if guessed_time:
                logging.debug(f"Guessed time for {url}: {guessed_time}")

            if not data.get("url") or not data.get("content"):
                raise ValueError("No content found")

            # 垃圾内容检测
            spam_detect_length = 300
            is_good = len(data["content"]) > spam_detect_length or not await classify_text(
                data["content"]
            )

            if not is_good:
                logging.warning(
                    f"Blocked content {len(data['content'])}:",
                    {"url": url, "content": data["content"][:spam_detect_length]},
                )
                raise ValueError(f"Blocked content {url}")

            # 分块存储
            chunks, chunk_positions = chunk_text(data["content"])
            web_contents[data["url"]] = {
                "chunks": chunks,
                "chunk_positions": chunk_positions,
                "title": data.get("title"),
            }

            # 加入知识库
            answer = await cherry_pick(
                question, data["content"], {}, context, url
            )
            all_knowledge.append(
                {
                    "question": f'What do expert say about "{question}"?',
                    "answer": answer,
                    "references": [data["url"]],
                    "type": "url",
                    "updated": (
                        format_date_based_on_type(datetime.fromisoformat(guessed_time), "full")
                        if guessed_time
                        else None
                    ),
                }
            )

            # 处理页面内链接
            for link in data.get("links") or []:
                nn = normalize_url(link[1])
                if not nn:
                    continue
                snippet = {"title": link[0], "url": nn, "description": link[0]}
                add_to_all_urls(snippet, all_urls, 0.1)

            # 处理图片
            if with_images and data.get("images"):
                for alt, img_url in data["images"].items():
                    img_obj = await process_image(img_url, context.tokenTracker)
                    if img_obj and not any(i["url"] == img_obj["url"] for i in image_objects):
                        image_objects.append(img_obj)

            return {"url": url, "result": response}

        except Exception as e:
            logging.error("Error reading URL:"+str({"url": url, "error": str(e)}))
            bad_urls.append(url)

            # 根据错误信息收集坏 hostname
            msg = str(e).lower()
            if any(
                    k in msg
                    for k in (
                            "couldn't resolve host name",
                            "could not be resolved",
                            "err_cert_common_name_invalid",
                            "err_connection_refused",
                    )
            ) or (
                    e.__class__.__name__ in ("ParamValidationError", "AssertionFailureError")
                    and ("domain" in msg or "resolve host name" in msg)
            ):
                hostname = ""
                try:
                    hostname = extract_url_parts(url).hostname
                except Exception as parse_e:
                    logging.error("Error parsing URL for hostname:", {"url": url, "error": str(parse_e)})
                if hostname:
                    bad_hostnames.append(hostname)
                    logging.debug(f"Added {hostname} to bad hostnames list")
            return None

        finally:
            # 无论成败，只要 url 非空就记入已访问
            if url:
                visited_urls.append(url)
                context.actionTracker.track_action(
                    {
                        "thisStep": {
                            "action": "visit",
                            "think": "",
                            "URL_target": [url],
                        }
                    }
                )

    url_results = await asyncio.gather(*[_process_single(u) for u in urls])
    valid_results = [r for r in url_results if r is not None]

    # ---------- 3. 根据 bad_hostnames 清理 all_urls ----------
    if bad_hostnames:
        for u in list(all_urls.keys()):
            if extract_url_parts(u).hostname in bad_hostnames:
                del all_urls[u]
                logging.warning(f"Removed {u} from all_urls because of bad hostname")

    return {"urlResults": valid_results, "success": len(valid_results) > 0}
