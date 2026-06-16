import argparse
import json
import os
from dataclasses import dataclass
from pprint import pprint

import aiofiles
from dotenv import load_dotenv
from pydantic import BaseModel

from tool.build_refs import build_references, build_image_references
from tool.code_sandbox import CodeSandbox
from tool.date_tools import format_date_range, format_date_based_on_type
from tool.error_analyzer import analyze_steps
from tool.evaluator import evaluate_question, evaluation_answer
from tool.finalizer import finalizeAnswer
from tool.image_tools import dedup_images_with_embeddings, filter_images
from tool.intent_reg import intent_query
from tool.jina_dedup import dedup_queries
from core.plugin_manager import get_plugin_instance
from tool.queryrewriter import rewrite_query
from tool.serp_cluster import serp_cluster
from tool.text_tools import remove_html_tags, choose_k, build_md_from_answer, repairMarkdownFootnotesOuter, \
    fixCodeBlockIndentation, convertHtmlTablesToMd, repair_markdown_final
from utils.action_tracker import ActionTracker
from utils.memory import MemoryManager
from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import MAX_QUERIES_PER_STEP, LANGUAGE_CODE, set_language, set_search_language_code, \
    build_agent_action_payload, MAX_REFLECT_PER_STEP, MAX_URLS_PER_STEP
from utils.token_tracker import TokenTracker
from utils.url_tool import *

load_dotenv()
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER")
STEP_SLEEP = float(os.getenv("STEP_SLEEP"))


@dataclass
class KnowledgeItem:
    question: str
    answer: str
    sourceCode: str = None
    type: Optional[str] = None
    updated: Optional[str] = None
    references: Optional[List[str]] = None


@dataclass
class BoostedSearchSnippet:
    freq_boost: float
    hostname_boost: float
    path_boost: float
    jina_rerank_boost: float
    final_score: float


def remove_extra_line_breaks(text: str) -> str:
    return re.sub(r'\n{2,}', '\n\n', text)


def build_msgs_from_knowledge(knowledge: List[KnowledgeItem]) -> List[dict]:
    messages = []
    for k in knowledge:
        user_content = k.question.strip()
        messages.append({"role": "user", "content": user_content})

        answer_parts = []
        if k.updated and k.type in {"url", "side-info"}:
            answer_parts.append(f"<answer-datetime>\n{k.updated}\n</answer-datetime>")
        if k.references and k.type == "url":
            answer_parts.append(f"<url>\n{k.references[0]}\n</url>")
        answer_parts.append(k.answer.strip())

        assistant_content = remove_extra_line_breaks("\n".join(answer_parts))
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


# from your_module import KnowledgeItem, build_msgs_from_knowledge, remove_extra_line_breaks
def compose_msgs(
        messages: List[Dict[str, str]],
        knowledge: List[Any],  # List[KnowledgeItem]
        question: str,
        final_answer_pip: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    :param messages:历史对话
    :param knowledge:获取的知识
    :param qustion:原问题
    :param final_answer_pip:当前需求
    将知识库消息在前，真实用户-助手交互在后，最后追加当前用户问题（含 reviewer 要求）。
    输出：
    证据（知识库）→ 历史对话 → 用户原问题 + 需求（含 reviewer 反馈）
    """
    # 1. 知识在前
    msgs = build_msgs_from_knowledge(knowledge) + messages

    # 2. 构造当前用户内容
    pip_part = ""
    if final_answer_pip:
        reviewer_blocks = "\n".join(
            f"<reviewer-{idx + 1}>\n{p}\n</reviewer-{idx + 1}>"
            for idx, p in enumerate(final_answer_pip)
        )
        pip_part = f"""
<answer-requirements>
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments."
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- Follow reviewer's feedback and improve your answer quality.
{reviewer_blocks}
</answer-requirements>"""

    user_content = f"{question}{pip_part}".strip()
    user_content = remove_extra_line_breaks(user_content)

    # 3. 追加到末尾
    msgs.append({"role": "user", "content": user_content})
    return msgs


def get_prompt(
        context: Optional[List[str]] = None,
        all_questions: Optional[List[str]] = None,
        all_keywords: Optional[List[str]] = None,
        allow_reflect: bool = True,
        allow_answer: bool = True,
        allow_read: bool = True,
        allow_search: bool = True,
        allow_coding: bool = True,
        knowledge: List[KnowledgeItem] = None,  # List[KnowledgeItem] 若已定义可替换
        all_urls=None,
        beast_mode: bool = False,
) -> Dict[str, Optional[List[str]]]:
    sections: List[str] = []
    action_sections: List[str] = []

    # 头部
    sections.append(
        f"Current date: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')}\n\n"
        "You are a deep research assistant. You are specialized in multistep reasoning.\n"
        "Using your best knowledge, conversation with the user and lessons learned, "
        "answer the user question with absolute certainty."
    )

    # 上下文
    if context:
        sections.append(
            f"""
You have conducted the following actions:
<context>
{chr(10).join(context)}
</context>
"""
        )

    # 动作片段
    ## url visit
    url_list = sort_select_urls(all_urls or [], max_urls=20)
    if allow_read and url_list:
        url_str = "\n".join(
            f"  - [idx={idx + 1}] [weight={item['score']:.2f}] \"{item['url']}\": \"{item['merged'][:50]}\""
            for idx, item in enumerate(url_list)
        )
        action_sections.append(
            f"""
<action-visit>
- Ground the answer with external web content
- Read full content from URLs and get the fulltext, knowledge, clues, hints for better answer the question.
- Must check URLs mentioned in <question> if any
- Choose and visit relevant URLs below for more knowledge. higher weight suggests more relevant:
<url-list>
{url_str}
</url-list>
</action-visit>
"""
        )

    ## search
    if allow_search:
        bad_req = (
            f"""
- Avoid those unsuccessful search requests and queries:
<bad-requests>
{chr(10).join(all_keywords)}
</bad-requests>
""".strip()
            if all_keywords
            else ""
        )
        action_sections.append(
            f"""
<action-search>
- Use vector data base to find relevant information
- If the evidence obtained is not comprehensive, or if faced with open questions, constantly search from multiple perspectives.
- Build a search request based on the deep intention behind the original question and the expected answer format
- Add another request if the original question covers multiple aspects or elements and one query is not enough, each request focus on one specific aspect of the original question
{bad_req}
</action-search>
"""
        )

    ## answer
    if allow_answer:
        action_sections.append(
            """
<action-answer>
- For greetings, casual conversation, general knowledge questions, answer them directly.
- If user ask you to retrieve previous messages or chat history, remember you do have access to the chat history, answer them directly.
- For all other questions, provide a verified answer.
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- If uncertain, use <action-reflect>
</action-answer>
"""
        )

    if beast_mode:
        action_sections.append(
            """
<action-answer>
🔥 ENGAGE MAXIMUM FORCE! ABSOLUTE PRIORITY OVERRIDE! 🔥

PRIME DIRECTIVE:
- DEMOLISH ALL HESITATION! ANY RESPONSE SURPASSES SILENCE!
- PARTIAL STRIKES AUTHORIZED - DEPLOY WITH FULL CONTEXTUAL FIREPOWER
- TACTICAL REUSE FROM PREVIOUS CONVERSATION SANCTIONED
- WHEN IN DOUBT: UNLEASH CALCULATED STRIKES BASED ON AVAILABLE INTEL!

FAILURE IS NOT AN OPTION. EXECUTE WITH EXTREME PREJUDICE! ⚡️
</action-answer>
"""
        )

    ## 反思
    if allow_reflect:
        action_sections.append(
            """
<action-reflect>
- Think slowly and planning lookahead. Examine <question>, <context>, previous conversation with users to identify knowledge gaps.
- Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer
</action-reflect>
"""
        )

    ## Coding
    if allow_coding:
        action_sections.append(
            """
<action-coding>
- This Python-based solution helps you handle programming tasks such as counting, filtering, transforming, sorting, regex extraction, and data analysis.
- Typical implementations may use Python’s standard libraries (e.g., re, collections, itertools) or data analysis libraries (e.g., pandas, numpy).
- Simply describe your problem in the "coding_issue" field. For small inputs, include actual example values; for larger datasets, specify variable names.
- No coding is required — experienced Python engineers will handle the implementation based on your description.
</action-coding>          
"""
        )

    # 4. 把动作片段拼到一起
    sections.append(
        f"""
Based on the current context, you must choose one of the following actions:
<actions>
{chr(10).join(action_sections)}
</actions>
"""
    )

    # 5. 尾部
    sections.append(
        "Think step by step, choose the action, then respond by matching the schema of that action."
    )

    return {
        "system": remove_extra_line_breaks("\n\n".join(sections)),
        "url_list": [u['url'] for u in url_list]
    }


async def update_references(this_step: dict, all_urls: Dict[str, dict]):
    log = get_logger("update_references")
    references = this_step.get("references", [])
    updated_refs = []

    for ref in references:
        url = ref.get("url")
        if not url:
            continue
        normalized_url = normalize_url(url)
        if not normalized_url:
            continue
        all_url_info = all_urls.get(normalized_url, {})
        exact_quote = (
                ref.get('exactQuote') or
                all_url_info.get('description') or
                all_url_info.get('title') or
                ''
        )
        # 字符串替换，保留字母、数字和空格
        exact_quote = re.sub(r'[^\w\s]', ' ', exact_quote, flags=re.UNICODE)
        exact_quote = re.sub(r'\s+', ' ', exact_quote).strip()
        updated_ref = {
            **ref,
            'exactQuote': exact_quote,
            'title': all_url_info.get('title', ''),
            'url': normalized_url,
            'dateTime': ref.get('dateTime') or all_url_info.get('date', ''),
        }
        updated_refs.append(updated_ref)
    this_step["references"] = updated_refs

    # 并发异步处理URL的dateTime
    tasks = [
        get_last_modified(ref['url'])
        for ref in this_step['references']
        if not ref.get('dateTime')
    ]
    results = await asyncio.gather(*tasks)
    # 填充dateTime
    result_idx = 0
    for ref in this_step['references']:
        if not ref.get('dateTime'):
            ref['dateTime'] = results[result_idx] or ''
            result_idx += 1

    log.debug('Updated references:', {'references': this_step['references']})


async def execute_search_queries(
        keywords_queries: List[Dict[str, Any]],
        context: Any,
        all_urls: Dict[str, Dict[str, Any]],
        web_contents,
        only_hostnames: Optional[List[str]] = None,
        search_provider: Optional[str] = None,
        meta: Optional[str] = None
):
    log = get_logger("execute_search_queries")
    uniq_q_only = [q['q'] for q in keywords_queries]
    new_knowledge = []
    searched_queries = []
    context.actionTracker.track_think(
        'search_for',
        LANGUAGE_CODE,
        {'keywords': ', '.join(uniq_q_only)},
    )

    utility_score = 0

    async def _search_single(query: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个查询并返回原始结果，避免并发修改共享状态。"""
        results = []
        old_query = query['q']
        q_local = dict(query)
        if only_hostnames and len(only_hostnames) > 0:
            q_local['q'] = f"{q_local['q']} site:{' OR site:'.join(only_hostnames)}"
        try:
            log.info('Search query:' + str({'query': q_local}))
            provider = search_provider or SEARCH_PROVIDER or "jina"
            search_plugin = get_plugin_instance("search", provider, config={})
            num = None if meta else 30
            resp = await search_plugin.search(
                q_local, domain="arxiv" if provider == "arxiv" else None, num=num, meta=meta, tracker=context.tokenTracker
            )
            # 统一适配不同插件的返回结构
            if 'response' in resp and 'results' in resp['response']:
                results = resp['response']['results']
            elif 'response' in resp and 'result' in resp['response']:
                results = resp['response']['result']
            elif 'data' in resp:
                results = resp['data']
            else:
                results = resp.get('response', {}).get('results', [])
            if not results:
                raise Exception('No results found')
        except Exception as e:
            log.error(f"{SEARCH_PROVIDER} search failed for query:" + str({'query': q_local, 'error': e}))
            if hasattr(e, 'status') and e.status == 401:
                raise Exception(f'Unauthorized {provider} API key')
            return {"success": False, "old_query": old_query}
        finally:
            await asyncio.sleep(STEP_SLEEP)

        min_results = []
        for r in results:
            url = normalize_url(r.get('url') or r.get('link'))
            if not url:
                continue
            min_results.append({
                'title': r.get('title'),
                'url': url,
                'description': r.get('description') if 'description' in r else r.get('snippet'),
                'weight': 1,
                'date': r.get('date'),
            })

        clusters = []
        try:
            clusters = await serp_cluster(min_results, context)
        except Exception as e:
            log.warning("serpCluster failed:" + str({"error": str(e)}))

        joined_desc = "; ".join([r.get("description") or "" for r in min_results])
        side_info = KnowledgeItem(
            question=f'What do Internet say about "{old_query}"?',
            answer=remove_html_tags(joined_desc),
            type="side-info",
            updated=format_date_range(q_local) if q_local.get("tbs") else None,
        )

        return {
            "success": True,
            "old_query": old_query,
            "min_results": min_results,
            "clusters": clusters,
            "side_info": side_info,
        }

    # 并行执行所有搜索查询
    search_results = await asyncio.gather(*[_search_single(q) for q in keywords_queries])

    for sr in search_results:
        if not sr["success"]:
            continue
        old_query = sr["old_query"]
        min_results = sr["min_results"]
        clusters = sr["clusters"]
        side_info = sr["side_info"]

        for r in min_results:
            utility_score += add_to_all_urls(r, all_urls)
            web_contents[r['url']] = {
                'title': r['title'],
                'chunks': [r['description']],
                'chunk_positions': [[0, len(r['description'] or '')]],
            }

        searched_queries.append(old_query)

        for c in clusters:
            new_knowledge.append(
                KnowledgeItem(
                    question=c.get("question"),
                    answer=c.get("insight"),
                    references=getattr(c, "urls", None),
                    type="url",
                )
            )

        new_knowledge.append(side_info)
        context.actionTracker.track_action({
            "thisStep": {
                "action": "search",
                "think": "",
                "search_requests": [old_query],
            }
        })

    if len(searched_queries) == 0:
        if only_hostnames and len(only_hostnames) > 0:
            log.warning(
                "No results found for queries: {} on hostnames: {}".format(
                    ", ".join(filter(None, uniq_q_only)),
                    ", ".join(only_hostnames),
                )
            )
            context.actionTracker.trackThink(
                "hostnames_no_results",
                LANGUAGE_CODE,
                {"hostnames": ", ".join(only_hostnames)},
            )
    else:
        log.debug(f"Utility/Queries: {utility_score}/{len(searched_queries)}")
        if len(searched_queries) > MAX_QUERIES_PER_STEP:
            log.debug('So many queries??? ' + ", ".join(f'"{q}"' for q in searched_queries))

    return {
        "newKnowledge": new_knowledge,
        "searchedQueries": searched_queries,
    }


class TrackerContext:
    def __init__(self):
        self.tokenTracker = TokenTracker()
        self.actionTracker = ActionTracker()


# 提前定义的辅助函数
def includes_eval(all_checks, eval_type) -> bool:
    return any(check["type"] == eval_type for check in all_checks)


def dedup_keywords(keywords_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 先按 q 分组
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for kq in keywords_queries:
        by_q.setdefault(kq["q"], []).append(kq)

    # 如果同一 q 出现多次，只保留裸 {q}；否则保留唯一那条
    def pick(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"q": matches[0]["q"]} if len(matches) > 1 else matches[0]

    return [pick(group) for group in by_q.values()]


async def get_response(
        question,
        search_languge_code,
        search_provider,
        with_images=False,
        token_budget=1000000,
        max_bad_attempts=2,
        existing_context=2,
        messages=None,
        num_returned_urls=100,
        no_direct_answer=False,
        boost_hostnames=None,
        bad_hostnames=None,
        only_hostnames=None,
        max_ref=10,
        min_rel_score=0.8,
        language_code=None,
        team_size=1
):
    step = 0  # 应该是0
    total_step = 0  # 应该是0
    all_context = []
    log = get_logger("get_response")

    def update_context(s):
        all_context.append(s)

    if messages is not None:
        messages = [m for m in messages if m.get("role") != 'system']
    question = question.strip() if question is not None else None
    if messages and len(messages) > 0:
        last_content = messages[-1].get("content")
        if isinstance(last_content, str):
            question = last_content.strip()
        elif isinstance(last_content, list):
            # 筛选出 type 为 'text' 的所有内容
            text_contents = [c for c in last_content if isinstance(c, dict) and c.get('type') == 'text']

            # 取最后一个（如果有），取其 'text' 字段，否则空字符串
            question = text_contents[-1]['text'] if text_contents else ''

    elif messages:
        messages = [{'role': 'user', 'content': question.strip()}]
    else:
        messages = []

    set_language(language_code or question)
    if search_languge_code is not None:
        set_search_language_code(search_languge_code)

    context = TrackerContext()
    context.tokenTracker = getattr(existing_context, 'tokenTracker',
                                   TokenTracker(token_budget)) if existing_context is not None else TokenTracker(
        token_budget)
    context.actionTracker = getattr(existing_context, 'actionTracker',
                                    ActionTracker()) if existing_context is not None else ActionTracker()

    generator = ObjectGeneratorSafe(context.tokenTracker)
    schema = build_agent_action_payload(True, True, True, True, True)
    gaps = [question]
    all_questions = [question]
    all_keywords = []
    candidate_answers = []
    # 记忆模块：统一管理检索知识，超出上限时自动压缩，每轮迭代读取
    memory = MemoryManager(make_item=KnowledgeItem, token_tracker=context.tokenTracker)
    all_knowledge = memory.items  # 与记忆模块共享同一列表引用
    weighted_urls = []

    diary_context = []

    clarification_questions = await intent_query(question, context)
    if clarification_questions:
        diary_context.append(
            f"Clarification questions suggested: {clarification_questions}"
        )


    allow_answer = False
    allow_read = True
    allow_search = True
    allow_reflect = True
    allow_coding = False
    msg_knowledge = []

    this_step = {
        'action': 'answer',
        'answer': '',
        'references': [],
        'think': '',
        'isFinal': False
    }

    all_URLs = {}
    all_web_contents = {}
    visited_URLs = []
    bad_URLs = []
    image_objects = []
    evaluation_metrics = {}
    regular_budget = token_budget * 0.85
    final_answer_PIP = []
    trivial_question = False

    for m in messages:
        str_msg = ''
        if isinstance(m.get("content"), str):
            str_msg = m.get("content").strip()
        elif isinstance(m.get("content"), dict) and isinstance(m.get("content"), list):

            str_msg = '\n'.join(
                c['text'] for c in m.get("content") if c.get('type') == 'text'
            ).strip()

        for u in extract_urls_with_description(str_msg):
            add_to_all_urls(u, all_URLs)
    while context.tokenTracker.get_total_usage().totalTokens < regular_budget and total_step < 50:
        step += 1
        total_step += 1
        # 每轮迭代先读取并整理记忆：若记忆超出上限，则压缩较早的知识，避免上下文膨胀
        try:
            compressed = await memory.consolidate(reason=f"step-{total_step}")
            if compressed:
                log.debug(f"Memory consolidated at step {total_step}, now {len(all_knowledge)} items")
        except Exception as e:
            log.warning(f"Memory consolidate failed: {e}")
        budget_percentage = f"{(context.tokenTracker.get_total_usage().totalTokens / token_budget * 100):.2f}"
        log.debug(f"Step {total_step} / Budget used {budget_percentage}%" + str({" gaps": gaps}))
        allow_reflect = allow_reflect and (len(gaps) <= MAX_REFLECT_PER_STEP)
        # 轮询取出当前问题
        current_question: str = gaps[total_step % len(gaps)]

        if current_question.strip() == question and total_step == 1:
            eval_types = await evaluate_question(question, context)
            evaluation_metrics[current_question] = [
                {"type": e, "numEvalsRequired": max_bad_attempts} for e in eval_types
            ]
            evaluation_metrics[current_question].append(
                {"type": "strict", "numEvalsRequired": max_bad_attempts}
            )
        elif current_question.strip() != question:
            # 非原始问题，初始化为空列表
            evaluation_metrics[current_question] = []

        if total_step == 1 and includes_eval(evaluation_metrics[current_question], "freshness"):
            # 在第一步检测到 freshness 时，禁止直接回答与反射
            allow_answer = False
            allow_reflect = False

        # 尚未测试，重排序+每个hostname保留top-2个
        if all_URLs and len(all_URLs) > 0:
            filtered = filter_urls(
                all_URLs,
                visited_URLs,
                bad_hostnames,
                only_hostnames
            )
            # rerank
            weighted_urls = await rank_urls(
                filtered,
                {
                    "question": current_question,
                    "boostHostnames": boost_hostnames
                },
                context
            )
            # 提升多样性：每个 hostname 最多留 top-2
            weighted_urls = keep_k_per_hostname(weighted_urls, 2)

            log.debug("Weighted URLs:" + str({" count": len(weighted_urls)}))

        allow_read = allow_read and len(weighted_urls) > 0
        allow_search = allow_search and len(weighted_urls) < 50  # disable search when too many urls already

        generate_prompt = get_prompt(
            diary_context,
            all_questions,
            all_keywords,
            allow_reflect,
            allow_answer,
            allow_read,
            allow_search,
            allow_coding,
            all_knowledge,
            weighted_urls,
            False
        )
        system = generate_prompt.get("system")
        url_list = generate_prompt.get("url_list")
        schema = build_agent_action_payload(
            allow_answer=allow_answer,
            allow_read=allow_read,
            allow_search=allow_search,
            allow_reflect=allow_reflect,
            allow_coding=allow_coding,
            current_question=current_question,
        )

        msg_with_knowledge = compose_msgs(
            messages,
            all_knowledge,
            current_question,
            final_answer_PIP if current_question == question else None
        )
        # print("total_step: ", total_step)
        # with open(f'test/msg_with_knowledge_{total_step}.json', 'w') as outfile:
        #     json.dump(json.dumps(msg_with_knowledge), outfile)
        # with open(f'test/system_{total_step}.json', 'w') as outfile:
        #     json.dump(generate_prompt, outfile)
        result = await generator.generate_object({
            "model": "agent",
            "schema": schema,
            "system": system,
            "messages": msg_with_knowledge,
            "numRetries": 2
        })
        obj = result.get("object", {}) if isinstance(result, dict) else {}
        action = obj.get("action")
        if obj.get(action) is None:
            print(result)
            continue
        this_step = {
            "action": action,
            "think": obj.get("think"),
            **obj.get(action)
        }
        actions = [allow_search, allow_read, allow_answer, allow_reflect, allow_coding]
        action_names = ['search', 'read', 'answer', 'reflect', 'coding']

        actions_str = ', '.join([name for allowed, name in zip(actions, action_names) if allowed])
        log.debug(f"`Step decision: {this_step['action']} <- [{actions_str}]`, {this_step}, {current_question}")
        context.actionTracker.track_action({
            "totalStep": total_step,
            "thisStep": this_step,
            "gaps": gaps,
        })
        # evaluation_metrics[current_question] = [{"type": "strict", "numEvalsRequired": max_bad_attempts}]

        allow_answer = True
        allow_read = True
        allow_search = True
        allow_reflect = True
        allow_coding = True
        if this_step.get("action") is not None and this_step["action"] == "answer":
            print("answer")
            if total_step == 1 and not no_direct_answer:
                this_step["isFinal"] = True
                trivial_question = True
                break
            update_context({
                "thisStep": this_step,
                'question': current_question,
                **this_step
            })
            log.debug('current question evaluation: ' + str({
                'question': current_question,
                'metrics': evaluation_metrics[current_question],
            }))

            evaluation = {
                "pass_": True,
                "think": ''
            }
            if evaluation_metrics.get(current_question) is not None and len(
                    evaluation_metrics.get(current_question)) > 0:
                context.actionTracker.track_think('eval_first', language_code)
                evaluation_types = [e.get("type") for e in evaluation_metrics.get(current_question) if
                                    e.get('numEvalsRequired') > 0]

                evaluation = await evaluation_answer(
                    current_question,
                    this_step,
                    evaluation_types,
                    context,
                    all_knowledge,
                )
            if current_question.strip() == question.strip():
                allow_coding = False

                if evaluation.get("pass_"):
                    diary_context.append(f"""
At step {step}, you took **answer** action and finally found the answer to the original question:

Original question: 
{current_question}

Your answer: 
{this_step['answer']}

The evaluator thinks your answer is good because: 
{evaluation['think']}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
""")
                    this_step["isFinal"] = True
                    break
                else:
                    for e in evaluation_metrics[current_question]:
                        if e.get("type") == evaluation.get("type"):
                            e["numEvalsRequired"] -= 1
                    # 然后过滤
                    evaluation_metrics[current_question] = [
                        e for e in evaluation_metrics[current_question] if e.get("numEvalsRequired") > 0
                    ]
                    if evaluation.get("type") == 'strict' and evaluation.get("improvement_plan"):
                        final_answer_PIP.append(evaluation["improvement_plan"])
                    if len(evaluation_metrics[current_question]) == 0:
                        this_step["isFinal"] = False
                        break
                    diary_context.append(f"""
At step {step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
{current_question}

Your answer: 
{this_step.get("answer")}

The evaluator thinks your answer is bad because: 
{evaluation.get("think")}
""")
                    error_analysis = await analyze_steps(diary_context, context)
                    all_knowledge.append(KnowledgeItem(
                        question=f"""Why is the following answer bad for the question? Please reflect

<question>
{current_question}
</question>

<answer>
{this_step.get("answer")}
</answer>
""",
                        answer=f"""
{evaluation.get('think')}

{error_analysis.get('recap')}

{error_analysis.get('blame')}

{error_analysis.get('improvement')}
""",
                        type='qa'))
                    allow_answer = False
                    diary_context = []
                    step = 0
            elif evaluation.get("pass_"):
                diary_context.append(f"""At step {step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
{current_question}

Your answer: 
{this_step.get("answer")}

The evaluator thinks your answer is good because: 
{evaluation.get('think')}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.""")
                all_knowledge.append(KnowledgeItem(
                    question=current_question,
                    answer=this_step["answer"],
                    type='qa',
                    updated=format_date_based_on_type(datetime.now(), 'full')
                ))
                if current_question in gaps:
                    gaps.remove(current_question)
        elif this_step['action'] == 'reflect' and this_step.get('question2answer'):
            print("reflect")
            this_step['question2answer'] = choose_k(
                (await dedup_queries(this_step['question2answer'], all_questions,
                                     context.tokenTracker)).get("unique_queries"),
                MAX_REFLECT_PER_STEP
            )
            new_gap_questions = this_step['question2answer']

            if new_gap_questions:
                # found new gap questions
                diary_context.append(f"""
            At step {step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "{current_question}"
            You realize you need to know the answers to the following sub-questions:
            {chr(10).join([f"- {q}" for q in new_gap_questions])}

            You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
            """)
                gaps.extend(new_gap_questions)
                all_questions.extend(new_gap_questions)
                update_context({
                    **this_step,
                    'total_step': total_step,
                })
            else:
                diary_context.append(f"""
            At step {step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "{current_question}" into gap-questions like this: {', '.join(new_gap_questions)} 
            But then you realized you have asked them before. You decided to think out of the box or cut from a completely different angle. 
            """)
                update_context({
                    **this_step,
                    'total_step': total_step,
                    'result': "You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!"
                })

            allow_reflect = False
        elif this_step['action'] == 'search' and this_step.get('search_requests'):
            print('search')
            this_step['search_requests'] = choose_k(
                (await dedup_queries(
                    this_step['search_requests'],
                    [],
                    context.tokenTracker)
                 ).get("unique_queries"),
                MAX_QUERIES_PER_STEP
            )
            esq_res = await execute_search_queries(
                [{"q": q} for q in this_step.get('search_requests', '')],
                context,
                all_URLs,
                all_web_contents,
                None,  # 对应 TS 的 undefined
                search_provider,
            )
            searched_queries, new_knowledge = esq_res.get("searchedQueries"), esq_res.get("newKnowledge")

            all_keywords.extend(searched_queries)
            all_knowledge.extend(new_knowledge)
            sound_bites = ' '.join(k.answer for k in new_knowledge)

            if team_size > 1:
                print("并行查询，暂时没有")
            keywords_queries = await rewrite_query(this_step, sound_bites, context)
            q_only = [q['q'] for q in keywords_queries]
            uniq_q_only = choose_k(
                (
                    await dedup_queries(
                        q_only,
                        all_keywords,
                        context.tokenTracker
                    )
                ).get("unique_queries"),
                MAX_QUERIES_PER_STEP
            )
            temp = []

            for q in uniq_q_only:
                matches = [kq for kq in keywords_queries if kq.get("q") == q]
                if len(matches) > 1:
                    temp.append({'q': q})
                elif matches:
                    temp.append(matches[0])
                else:
                    temp.append({'q': q})
            keywords_queries = temp

            any_result = False

            if len(keywords_queries) > 0:
                esq_res = await execute_search_queries(
                    keywords_queries,
                    context,
                    all_URLs,
                    all_web_contents,
                    only_hostnames,
                    search_provider,
                )
                searched_queries, new_knowledge = esq_res.get("searchedQueries"), esq_res.get("newKnowledge")

                if len(searched_queries) > 0:
                    any_result = True
                    all_keywords.extend(searched_queries)
                    all_knowledge.extend(new_knowledge)
                    diary_context.append(f"""
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: "{", ".join(str(item["q"]) for item in keywords_queries)}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
""")
                    update_context({
                        'total_step': total_step,
                        'question': current_question,
                        'result': result,
                        **this_step
                    })
            if not any_result or not keywords_queries:
                diary_context.append(
                    f"""
At step {step}, you took the **search** action and looked for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: "{', '.join(str(item['q']) for item in keywords_queries)}".
But then you realized you have already searched for these keywords before; no new information was returned.
You decided to think out of the box or cut from a completely different angle.
""")
                update_context({
                    'total_step': total_step,
                    'result': "You have tried all possible queries and found no new information. You must think out of the box or different angle!!!",
                    **this_step
                })
            allow_search = False
            allow_answer = False
        elif this_step['action'] == 'visit' and this_step.get('URL_target') and len(url_list) > 0:
            print('visit')
            this_step['URL_target'] = [
                normalize_url(url_list[idx - 1])
                for idx in (this_step.get('URL_target') or [])  # 等价于 (thisStep.URLTargets as number[])
            ]
            step_url_targets = [
                url for url in this_step['URL_target']
                if url and url not in visited_URLs
            ]
            weighted_urls_list = [r['url'] for r in weighted_urls if r.get('url')]
            this_step['URL_target'] = list(dict.fromkeys(step_url_targets + weighted_urls_list))[:MAX_URLS_PER_STEP]
            unique_URLs = this_step.get("URL_target")
            log.debug('Unique URLs: ' + str(unique_URLs))
            if len(unique_URLs) > 0:
                pu = await process_urls(
                    unique_URLs,
                    context,
                    all_knowledge,
                    all_URLs,
                    visited_URLs,
                    bad_URLs,
                    image_objects,
                    current_question,
                    all_web_contents,
                    with_images
                )
                url_results, success = pu.get("urlResults"), pu.get("success")
                _visited_urls_str = "\n".join(
                    r["url"] for r in url_results if r is not None
                )
                diary_context.append(
                    f"""At step {step}, you took the **visit** action and deep dive into the following URLs:
{_visited_urls_str}
You found some useful information on the web and add them to your knowledge for future reference.""" if success else f"At step {step}, you took the **visit** action and try to visit some URLs but failed to read the content. You need to think out of the box or cut from a completely different angle."
                )
                if success:
                    update_context({
                        'total_step': total_step,
                        'question': current_question,
                        **this_step,
                        'result': url_results
                    })
                else:
                    update_context({
                        'total_step': total_step,
                        **this_step,
                        'result': 'You have tried all possible URLs and found no new information. You must think out of the box or different angle!!!'
                    })

            else:
                diary_context.append(f"""
At step {step}, you took the **visit** action. But then you realized you have already visited these URLs and you already know very well about their contents.
You decided to think out of the box or cut from a completely different angle.""")
                update_context({
                    'total_step': total_step,
                    **this_step,
                    'result': 'You have visited all possible URLs and found no new information. You must think out of the box or different angle!!!'
                })

            allow_read = False
        elif this_step.get("action") == 'coding' and this_step.get("coding_issue"):
            print('coding')
            sandbox = CodeSandbox(
                {
                    "allContext": all_context,
                    "URLs": weighted_urls[:20],
                    "allKnowledge": all_knowledge,
                },
                context,
            )
            try:
                result = await sandbox.solve(this_step['coding_issue'])
                solution = result["solution"]
                attempts = result["attempts"]
                all_knowledge.append(
                    KnowledgeItem(
                        question=f"What is the solution to the coding issue: {this_step['coding_issue']}?",
                        answer=str(solution["output"]),
                        sourceCode=solution["code"],
                        type='coding',
                        updated=format_date_based_on_type(datetime.now(), 'full')
                    )
                )
                diary_context.append(f"""
At step {step}, you took the **coding** action and try to solve the coding issue: {this_step['coding_issue']}.
You found the solution and add it to your knowledge for future reference.
""")
                update_context({
                    'total_step': total_step,
                    'result': result,
                    **this_step
                })
            except Exception as e:
                log.error("Error solving coding issue:" + str({
                    'error': e if isinstance(e, str) else str(e),
                }))
                diary_context.append(f"""
At step {step}, you took the **coding** action and try to solve the coding issue: {this_step['coding_issue']}.
But unfortunately, you failed to solve the issue. You need to think out of the box or cut from a completely different angle.
""")
                update_context({
                    'total_step': total_step,
                    'result': 'You have tried all possible solutions and found no new information. You must think out of the box or different angle!!!',
                    **this_step
                })
            finally:
                allow_read = False

        await store_context(
            system,
            schema,
            {
                'allContext': all_context,
                'allKeywords': all_keywords,
                'allQuestions': all_questions,
                'allKnowledge': all_knowledge,
                'weightedURLs': weighted_urls,
                'msgWithKnowledge': msg_with_knowledge,
            },
            total_step
        )
        # break
        await asyncio.sleep(STEP_SLEEP)

    if not this_step.get("isFinal", False):
        # 计算 token 使用百分比
        total_usage = context.tokenTracker.get_total_usage()
        percent = (total_usage.totalTokens / token_budget) * 100
        log.info(
            f"Beast mode!!! budget {percent:.2f}%" +
            str({
                "usage": context.tokenTracker.get_total_usage_snake_case(),
                "evaluationMetrics": evaluation_metrics,
                "maxBadAttempts": max_bad_attempts,
            }),
        )
        step += 1
        total_step += 1
        system = get_prompt(
            diary_context,
            all_questions,
            all_keywords,
            False,
            False,
            False,
            False,
            False,
            all_knowledge,
            weighted_urls,
            True,
        )["system"]
        schema = build_agent_action_payload(False, False, False, True, False, question)
        msg_with_knowledge = compose_msgs(messages, all_knowledge, question, final_answer_PIP)

        result = await generator.generate_object({
            "model": "agentBeastMode",
            "schema": schema,
            "system": system,
            "messages": msg_with_knowledge,
            "numRetries": 2,
        })
        obj = result.get("object", {}) if isinstance(result, dict) else {}
        this_step = {
            "action": obj.get("action"),
            "think": obj.get("think"),
            **obj.get(obj.get("action"), {}),
        }
        # thisStep 视为 AnswerAction
        this_step["isFinal"] = True

        # 跟踪动作
        context.actionTracker.track_action({
            "totalStep": total_step,
            "thisStep": this_step,
            "gaps": gaps
        })

    answer_step = this_step
    if trivial_question:
        answer_step["mdAnswer"] = build_md_from_answer(answer_step)

    elif not answer_step.get("isAggregated"):
        # 处理答案：修复 Markdown、URL、代码块、脚注
        finalized_answer = await finalizeAnswer(
            answer_step["answer"],
            all_knowledge,
            context
        )

        repaired_answer = repairMarkdownFootnotesOuter(
            fixCodeBlockIndentation(
                fix_bad_url_md_links(
                    convertHtmlTablesToMd(finalized_answer),
                    all_URLs
                )
            )
        )

        answer_step["answer"] = repair_markdown_final(repaired_answer)

        # 生成引用
        result = await build_references(
            answer_step["answer"],
            all_web_contents,
            context,
            80,
            max_ref,
            min_rel_score,
            only_hostnames
        )

        answer_step["answer"] = result["answer"]
        answer_step["references"] = result["references"]

        await update_references(answer_step, all_URLs)
        answer_step["mdAnswer"] = repairMarkdownFootnotesOuter(build_md_from_answer(answer_step))

        # 图片引用逻辑
        if image_objects and with_images:
            try:
                image_refs = await build_image_references(
                    answer_step["answer"],
                    image_objects,
                    context
                )
                answer_step["imageReferences"] = image_refs

                log.debug(
                    "Image references built:",
                    {
                        "imageReferences": [
                            {"url": i["url"], "score": i["relevanceScore"], "answerChunk": i["answerChunk"]}
                            for i in image_refs
                        ]
                    }
                )
            except Exception as error:
                log.error("Error building image references:", {"error": str(error)})
                answer_step["imageReferences"] = []

    else:
        # 聚合模式：合并答案
        answer_step["answer"] = "\n\n".join(candidate_answers)
        result = await build_references(
            answer_step["answer"],
            all_web_contents,
            context,
            80,
            max_ref,
            min_rel_score,
            only_hostnames
        )
        answer_step["answer"] = result["answer"]
        answer_step["references"] = result["references"]
        await update_references(answer_step, all_URLs)
        # answerStep["answer"] = await reduceAnswers(candidateAnswers, context, SchemaGen)
        answer_step["mdAnswer"] = repairMarkdownFootnotesOuter(build_md_from_answer(answer_step))

        # if with_images and answer_step.get("imageReferences"):
        #     sorted_images = sorted(
        #         answer_step["imageReferences"],
        #         key=lambda img: img.get("relevanceScore", 0),
        #         reverse=True
        #     )
        #
        #     log.debug("[agent] all sorted image references:", {"count": len(sorted_images)})
        #
        #     deduped = dedup_images_with_embeddings(sorted_images, [])
        #     filtered = filter_images(sorted_images, deduped)
        #
        #     log.debug("[agent] filtered images:", {"count": len(filtered)})
        #
        #     # 限制最多 10 张图像
        #     answer_step["imageReferences"] = filtered[:10]

    returned_urls = [r["url"] for r in weighted_urls[:num_returned_urls] if r and r.get("url")]
    return {
        "result": this_step,
        "context": context,
        "visitedURLs": returned_urls,  # deprecated
        "readURLs": [url for url in visited_URLs if url not in bad_URLs],
        "allURLs": [r["url"] for r in weighted_urls],
        "imageReferences": this_step['image_references'] if with_images else None,
    }


def zod2json_schema(schema):
    """
    将 Pydantic BaseModel 转换为 JSON Schema 格式。
    若不是 Pydantic 模型，则直接返回原对象。
    """
    # 类或实例统一判断
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        # 是类
        return schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
    if isinstance(schema, BaseModel):
        # 是实例
        return schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
    return schema


def safe_json(obj):
    """安全 JSON 序列化：如果 obj 为空，返回 'null'"""
    if not obj and obj != 0:  # 排除数值 0
        return "null"
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as e:
        logging.warning(f"Failed to serialize object: {e}")
        return "null"


def _json_default(o):
    """处理自定义对象的 JSON 转换"""
    if hasattr(o, "dict"):  # 支持 pydantic 模型
        return o.dict()
    if hasattr(o, "__dict__"):  # 普通类实例
        return o.__dict__
    if hasattr(o, "__slots__"):  # 使用 __slots__ 的类
        return {k: getattr(o, k) for k in o.__slots__}
    return str(o)


async def store_context(prompt, schema, memory, step):
    """
    Python 等价版本的 storeContext()
    :param prompt: str
    :param schema: 任意对象 (Python dict)
    :param memory: dict，包含 allContext, allKeywords, allQuestions, allKnowledge, weightedURLs, msgWithKnowledge
    :param step: int
    """
    schema = zod2json_schema(schema)

    dir_path = f'./store_context/{step}/'
    os.makedirs(dir_path, exist_ok=True)

    allContext = memory.get('allContext')
    allKeywords = memory.get('allKeywords')
    allQuestions = memory.get('allQuestions')
    allKnowledge = memory.get('allKnowledge')
    weightedURLs = memory.get('weightedURLs')
    msgWithKnowledge = memory.get('msgWithKnowledge')

    async def _write_file(filename, content):
        try:
            async with aiofiles.open(dir_path + filename, "w", encoding="utf-8") as f:
                await f.write(content)
        except Exception as error:
            logging.error(f"Context storage failed for {filename}: {error}")

    prompt_content = f"""
Prompt:
{prompt}

JSONSchema:
{safe_json(schema)}
"""

    await asyncio.gather(
        _write_file(f"prompt-{step}.txt", prompt_content),
        _write_file("context.json", safe_json(allContext)),
        _write_file("queries.json", safe_json(allKeywords)),
        _write_file("questions.json", safe_json(allQuestions)),
        _write_file("knowledge.json", safe_json(allKnowledge)),
        _write_file("urls.json", safe_json(weightedURLs)),
        _write_file("messages.json", safe_json(msgWithKnowledge)),
    )


async def main():
    parser = argparse.ArgumentParser(description="Run get_response with command-line arguments.")

    parser.add_argument("--question", type=str, required=True, help="User question input")
    parser.add_argument("--search_language_code", type=str, default="en", help="Language code for search")
    parser.add_argument("--search_provider", type=str, default="jina", help="Search provider (e.g. jina, none)")
    parser.add_argument("--language_code", type=str, default="en", help="Language code for response")
    parser.add_argument("--with_images", action="store_true", help="Enable image analysis")
    parser.add_argument("--token_budget", type=int, default=10000000, help="Maximum token budget")
    parser.add_argument("--max_bad_attempts", type=int, default=2, help="Number of bad attempts before stopping")
    parser.add_argument("--existing_context", type=str, default=None, help="Existing tracker context (if any)")
    parser.add_argument("--num_returned_urls", type=int, default=5, help="Number of URLs to return")
    parser.add_argument("--no_direct_answer", action="store_true", help="Disable direct answer mode")
    parser.add_argument("--boost_hostnames", nargs="*", default=[], help="Hostnames to boost")
    parser.add_argument("--bad_hostnames", nargs="*", default=[], help="Hostnames to penalize")
    parser.add_argument("--only_hostnames", nargs="*", default=None, help="Restrict to specific hostnames")
    parser.add_argument("--max_ref", type=int, default=50, help="Maximum reference count")
    parser.add_argument("--min_rel_score", type=float, default=0.7, help="Minimum relevance score")
    parser.add_argument("--team_size", type=int, default=1, help="Team size for parallel processing")

    args = parser.parse_args()


    # 处理 existing_context：命令行传入的是字符串，需反序列化或置为 None
    existing_ctx = None
    if args.existing_context:
        try:
            existing_ctx = json.loads(args.existing_context)
        except Exception:
            existing_ctx = None

    # 调用 get_response
    result = await get_response(
        question=args.question,
        search_languge_code=args.search_language_code,
        search_provider=args.search_provider if args.search_provider.lower() != "none" else None,
        language_code=args.language_code,
        with_images=args.with_images,
        token_budget=args.token_budget,
        max_bad_attempts=args.max_bad_attempts,
        existing_context=existing_ctx,
        messages=[],
        num_returned_urls=args.num_returned_urls,
        no_direct_answer=args.no_direct_answer,
        boost_hostnames=args.boost_hostnames,
        bad_hostnames=args.bad_hostnames,
        only_hostnames=args.only_hostnames,
        max_ref=args.max_ref,
        min_rel_score=args.min_rel_score,
        team_size=args.team_size,
    )

    with open("result_output.txt", "a", encoding="utf-8") as f:
        f.write(str(result) + "\n\n")

    pprint(result.get("result"))


if __name__ == "__main__":
    asyncio.run(main())
