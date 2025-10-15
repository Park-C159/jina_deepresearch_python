import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

from utils.url_tool import *


@dataclass
class KnowledgeItem:
    question: str
    answer: str
    type: Optional[str] = None
    updated: Optional[str] = None
    references: Optional[List[str]] = None


@dataclass
class BoostedSearchSnippet:
    url: str
    merged: str
    score: float


def remove_extra_line_breaks(text: str) -> str:
    return re.sub(r'\n\s*\n+', '\n\n', text.strip())


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

async def update_references(this_step:dict, all_urls: Dict[str, dict]):
    references = this_step.get("reference", [])
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
    this_step["reference"] = updated_refs

    # 并发异步处理URL的dateTime
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

    logging.debug('Updated references:', {'references': this_step['references']})


if __name__ == "__main__":
    prompt = get_prompt(
        context=["searched: 'life meaning'", "visited: https://example.com/life"],
        all_keywords=["meaning of life", "purpose philosophy"],
        allow_search=True,
        beast_mode=True,
    )
    # 测试数据
    this_step = {
        "reference": [
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
    all_urls = {
        "example.com/news/123": {
            "title": "First Example News",
            "description": "Describes amazing event.",
            "date": "2022-05-01T08:00:00Z"
        },
        "another-site.org/info": {
            "title": "Another Site Info",
            "description": "Information page.",
            "date": "2023-09-09T09:00:00Z"
        },
        "example.com/news/456": {
            "title": "Second News",
            "description": "Update on previous news!",
            "date": "2024-05-05T12:00:00Z"
        }
    }
    update_references(this_step, all_urls)
    from pprint import pprint
    pprint(this_step)

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
