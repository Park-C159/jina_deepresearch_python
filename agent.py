import os
from dataclasses import dataclass
from pprint import pprint
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv

from tool.date_tools import format_date_range, format_date_based_on_type
from tool.error_analyzer import analyze_steps
from tool.evaluator import evaluate_question, evaluation_answer
from tool.jina_dedup import dedup_queries
from tool.jina_search import search
from tool.queryrewriter import rewrite_query
from tool.serp_cluster import serp_cluster
from tool.text_tools import remove_html_tags, choose_k
from utils.action_tracker import ActionTracker
from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import MAX_QUERIES_PER_STEP, LANGUAGE_CODE, set_langugae, set_search_language_code, \
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


def sort_select_urls(urls: List[BoostedSearchSnippet], top_k: int) -> List[BoostedSearchSnippet]:
    """按 score 降序选前 top_k 个"""
    return sorted(urls, key=lambda x: x.score, reverse=True)[:top_k]


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
        all_urls: Optional[List[BoostedSearchSnippet]] = None,
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
    url_list = sort_select_urls(all_urls or [], top_k=20)
    if allow_read and url_list:
        url_str = "\n".join(
            f"  - [idx={idx + 1}] [weight={item.score:.2f}] \"{item.url}\": \"{item.merged[:50]}\""
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
- Simply describe your problem in the "codingIssue" field. For small inputs, include actual example values; for larger datasets, specify variable names.
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
        "url_list": [u.url for u in url_list],
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

    for query in keywords_queries:
        results = []
        old_query = query['q']
        if only_hostnames and len(only_hostnames) > 0:
            query['q'] = f"{query['q']} site:{' OR site:'.join(only_hostnames)}"
        try:
            log.debug('Search query:', {'query': query})
            provider = search_provider or SEARCH_PROVIDER
            if provider in ('jina', 'arxiv'):
                num = None if meta else 30
                resp = search(query, num=num, meta=meta, tracker=context.tokenTracker)
                results = resp['response']['results'] if 'response' in resp and 'results' in resp['response'] else []
            ## 暂时不支持下面注释的网页搜索模式
            # elif provider == 'duck':
            #     resp = await duck_search(query['q'], {'safe_search': SafeSearchType.STRICT})
            #     results = resp['results'] if 'results' in resp else []
            # elif provider == 'brave':
            #     resp = await brave_search(query['q'])
            #     results = resp['response']['web']['results'] if 'response' in resp and 'web' in resp['response'] and 'results' in resp['response']['web'] else []
            # elif provider == 'serper':
            #     resp = await serper_search(query)
            #     results = resp['response']['organic'] if 'response' in resp and 'organic' in resp['response'] else []
            else:
                results = []
            if not results:
                raise Exception('No results found')
        except Exception as e:
            log.error(f"{SEARCH_PROVIDER} search failed for query:" + str({'query': query, 'error': e}))

            # 401 错误时中止
            if hasattr(e, 'response') and hasattr(e.response, 'status') and e.response.status == 401 and provider in (
                    'jina', 'arxiv'):
                raise Exception('Unauthorized Jina API key')
            continue
        finally:
            await asyncio.sleep(STEP_SLEEP)

        # 构建 minResults 列表
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

        for r in min_results:
            utility_score += add_to_all_urls(r, all_urls)
            web_contents[r['url']] = {
                'title': r['title'],
                # 'full': r['description'],
                'chunks': [r['description']],
                'chunk_positions': [[0, len(r['description'] or '')]],
            }

        searched_queries.append(query['q'])

        try:
            clusters = await serp_cluster(min_results, context)

            for c in clusters:
                new_knowledge.append(
                    KnowledgeItem(
                        question=c.get("question"),
                        answer=c.get("insight"),
                        references=getattr(c, "urls", None),
                        type="url",
                    )
                )
        except Exception as e:
            log.warning("serpCluster failed:" + str({"error": str(e)}))
        finally:
            joined_desc = "; ".join([r.get("description") or "" for r in min_results])
            side_info = KnowledgeItem(
                question=f'What do Internet say about "{old_query}"?',
                answer=remove_html_tags(joined_desc),
                type="side-info",
                updated=format_date_range(query) if query.get("tbs") else None,
            )
            new_knowledge.append(side_info)

            context.actionTracker.track_action({
                "thisStep": {
                    "action": "search",
                    "think": "",
                    "searchRequests": [old_query],
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
    total_step = 1  # 应该是0
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
        elif isinstance(last_content, dict) and isinstance(last_content, list):
            # 筛选出 type 为 'text' 的所有内容
            text_contents = [c for c in last_content if c.get('type') == 'text']

            # 取最后一个（如果有），取其 'text' 字段，否则空字符串
            question = text_contents[-1]['text'] if text_contents else ''
    elif messages:
        messages = [{'role': 'user', 'content': question.strip()}]
    else:
        messages = []

    set_langugae(language_code or question)
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
    all_knowledge = []

    diary_context = []
    weight_URLs = []

    allow_answer = True
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
    all_web_contents = []
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
    while context.tokenTracker.get_total_usage().totalTokens < regular_budget:
        step += 1
        total_step += 1
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
            weighted_urls = rank_urls(
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

        allow_read = allow_read and len(weight_URLs) > 0
        allow_search = allow_search and len(weight_URLs) < 50  # disable search when too many urls already

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
            weight_URLs,
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

        result = await generator.generate_object({
            "model": "agent",
            "schema": schema,
            "system": system,
            "messages": msg_with_knowledge,
            "numRetries": 2
        })
        action = None
        for act_key, act in result.get("object").get("action").items():
            if act is not None:
                action = act_key
        this_step = {
            "action": action,
            "think": result.get("object").get("think"),
        }
        actions = [allow_search, allow_read, allow_answer, allow_reflect, allow_coding]
        action_names = ['search', 'read', 'answer', 'reflect', 'coding']

        actions_str = ', '.join([name for allowed, name in zip(actions, action_names) if allowed])
        log.debug(f"`Step decision: {this_step["action"]} <- [{actions_str}]`, {this_step}, {current_question}")
        context.actionTracker.track_action({
            "totalStep": total_step,
            "thisStep": this_step,
            "gaps": gaps,
        })
        allow_answer = True
        allow_reflect = True
        allow_search = True
        allow_read = True
        allow_coding = True
        this_step["action"] = 'answer'
        this_step["answer"] = "OpenAI is a big company"
        evaluation_metrics[current_question] = [{"type": "strict", "numEvalsRequired": max_bad_attempts}]
        if this_step.get("action") is not None and this_step["action"] == "answer":
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
                "pass": True,
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

                if evaluation.get("pass"):
                    diary_context.append(f"""
At step {step}, you took **answer** action and finally found the answer to the original question:

Original question: 
{current_question}

Your answer: 
${this_step['answer']}

The evaluator thinks your answer is good because: 
${evaluation['think']}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
""")
                    this_step["isFinal"] = True
                    break
                else:
                    for e in evaluation_metrics[current_question]:
                        if e.type == evaluation.type:
                            e.numEvalsRequired -= 1
                    # 然后过滤
                    evaluation_metrics[current_question] = [
                        e for e in evaluation_metrics[current_question] if e.numEvalsRequired > 0
                    ]
                    if evaluation.get("type") == 'strict' and evaluation.get("improvement_plan"):
                        final_answer_PIP.append(evaluation["improvement_plan"])
                    if len(evaluation_metrics[current_question]) == 0:
                        this_step["isFinal"] = False
                        break
                    diary_context.append(f"""
At step ${step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
${current_question}

Your answer: 
${this_step.get("answer")}

The evaluator thinks your answer is bad because: 
${evaluation.get("think")}
""")
                    error_analysis = await analyze_steps({
                        diary_context,
                        context
                    })
                    all_knowledge.append({
                        'question': f"""Why is the following answer bad for the question? Please reflect

<question>
{current_question}
</question>

<answer>
${this_step.get("answer")}
</answer>
`,
            answer: `
${evaluation.get('think')}

${error_analysis.get('recap')}

${error_analysis.get('blame')}

${error_analysis.get('improvement')}
""",
                        'type': 'qa'
                    })
                    allow_answer = False
                    diary_context = []
                    step = 0
            elif evaluation.get("pass"):
                diary_context.append(f"""At step ${step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
{current_question}

Your answer: 
{this_step.get("answer")}

The evaluator thinks your answer is good because: 
{evaluation.get('think')}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.""")
                all_knowledge.append({
                    'question': current_question,
                    'answer': this_step["answer"],
                    'type': 'qa',
                    'updated': format_date_based_on_type(datetime.now(), 'full')
                })
                if current_question in gaps:
                    gaps.remove(current_question)
        elif this_step['action'] == 'reflect' and this_step.get('question2answer'):
            this_step['question2answer'] = choose_k(
                (await dedup_queries(this_step['question2answer'], all_questions,
                                     context.tokenTracker)).unique_queries,
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
                    **this_step.__dict__,
                    'total_step': total_step,
                })
            else:
                diary_context.append(f"""
            At step {step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "{current_question}" into gap-questions like this: {', '.join(new_gap_questions)} 
            But then you realized you have asked them before. You decided to think out of the box or cut from a completely different angle. 
            """)
                update_context({
                    **this_step.__dict__,
                    'total_step': total_step,
                    'result': "You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!"
                })

            allow_reflect = False
        elif this_step['action'] == 'search' and this_step.get('searchRequests'):
            this_step['searchRequests'] = choose_k(
                (await dedup_queries(
                    this_step['searchRequests'],
                    [],
                    context.tokenTracker)
                 ).unique_queries,
                MAX_QUERIES_PER_STEP
            )
            searched_queries, new_knowledge = await execute_search_queries(
                [{"q": q} for q in this_step.get('searchRequests', '')],
                context,
                all_URLs,
                all_web_contents,
                None,  # 对应 TS 的 undefined
                search_provider,
            )
            all_keywords.extend(searched_queries)
            all_knowledge.extend(new_knowledge)
            sound_bites = ' '.join(k['answer'] for k in new_knowledge)

            if team_size > 1:
                print("并行查询，暂时没有")
            keywords_queries = await rewrite_query(this_step, sound_bites, context)
            q_only = [q['q'] for q in keywords_queries]
            uniq_q_only = choose_k(
                (
                    await dedup_queries(
                        q_only,
                        all_knowledge,
                        context.tokenTracker
                    )
                ).unique_queries,
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
                searched_queries, new_knowledge = await execute_search_queries(
                    keywords_queries,
                    context,
                    all_URLs,
                    all_web_contents,
                    only_hostnames,
                    search_provider,
                )

                if len(searched_queries) > 0:
                    any_result = True
                    all_keywords.extend(searched_queries)
                    all_knowledge.extend(new_knowledge)
                    diary_context.append(f"""
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: "${", ".join(str(item["q"]) for item in keywords_queries)}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
""")
                    update_context({
                        'total_step': total_step,
                        'question': current_question,
                        'result': result,
                        **this_step.__dict__
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
                    **this_step.__dict__
                })
            allow_search = False
            allow_answer = False
        elif this_step['action'] == 'visit' and this_step.get('URLTargets') and len(url_list) > 0:
            this_step['URLTargets'] = [
                normalize_url(url_list[idx - 1])
                for idx in (this_step.get('URLTargets') or [])  # 等价于 (thisStep.URLTargets as number[])
            ]
            step_url_targets = [
                url for url in this_step['URLTargets']
                if url and url not in visited_URLs
            ]
            weighted_urls_list = [r['url'] for r in weighted_urls if r.get('url')]
            this_step['URLTargets'] = list(dict.fromkeys(step_url_targets + weighted_urls_list))[:MAX_URLS_PER_STEP]
            unique_URLs = this_step.get("URLTargets")
            log.debug('Unique URLs: ' + str(unique_URLs))
            if len(unique_URLs) > 0:
                url_results, success = process_urls(
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

            allow_read = False
        elif this_step.get("action") == 'coding' and this_step.get("codingIssue"):

            break
                
                
async def test(keywords_queries, context, all_urls, web_contents, only_hostnames=None):
    await get_response(
        "What is the current market value of OpenAI company?",
        "zh",
        "jina",
        # messages=[{
        #     'role': 'user',
        #     'content': "请访问www.example.com",
        # }],
        language_code="zh"
    )
    # pprint(result)


if __name__ == "__main__":
    # 测试数据
    web_contents = {
        "references": [
            {
                "url": "https://example.com/news/123",
                "exactQuote": None,
                "dateTime": None,
            },
            {
                "url": "http://another-site.org/info",
                # 已有 exactQuote
                "exactQuote": "This is an important fact!",
                # 已有 dateTime
                "dateTime": "2023-10-10T09:20:00Z",
            },
            {
                "url": "https://example.com/news/456",
                # exactQuote 不存在
            },
        ]
    }
    all_urls = {}
    keywords_queries = [
        {"q": "苹果最新的M系列芯片，都有哪些优点？"},
        {"q": "英伟达最新芯片是什么？有哪些优缺点？"},
    ]


    class TrackerContext:
        def __init__(self):
            self.tokenTracker = TokenTracker()
            self.actionTracker = ActionTracker()


    context = TrackerContext()
    asyncio.run(test(keywords_queries, context, all_urls, web_contents))

    # print(prompt["system"])

# if __name__ == "__main__":
#     knowledge_items = [
#         KnowledgeItem(
#             question="What is the capital of France?",
#             answer="Paris is the capital of France.",
#             type="url",
#             updated="2025-10-14",
#             references=["https://example.com/france"]
#         )
#     ]
#     pip = [
#         "Add more real-world examples.",
#         "Include counter-intuitive insights."
#     ]
#     history = [  # 已有的 CoreMessage
#         {"role": "user", "content": "What is life?"},
#         {"role": "assistant", "content": "Life is..."}
#     ]
#     full_msgs = compose_msgs(history, knowledge_items, "How to live a meaningful life?", pip)
#     for m in full_msgs:
#         print(m)

# msgs = build_msgs_from_knowledge(knowledge_items)
# for msg in msgs:
#     print(msg)
