"""
记忆模块 (Memory Manager)
=========================
为深度检索智能体提供"记忆"管理能力：

1. 统一存放检索过程中积累的知识条目（KnowledgeItem 等具备 question/answer 的对象）。
2. 估算记忆占用的 token 数；当超出 ``max_tokens`` 上限时，调用 LLM 把较早的记忆
   压缩成一条简洁但信息完整的摘要，保留关键事实 / 数据 / 来源线索，最近若干条保持原样。
3. 每轮迭代都可调用 ``consolidate()`` 读取并整理记忆，避免上下文无限膨胀、超出模型窗口。

设计要点：
- 与 KnowledgeItem 解耦：通过 ``make_item`` 工厂回调创建摘要条目，避免与 agent 循环导入。
- 原地维护内部列表（``self.items[:] = ...``），使外部持有同一列表引用者也能看到压缩结果。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, List, Optional

from utils.get_log import get_logger
from utils.safe_generator import ai_generate_object, count_text_tokens
from utils.token_tracker import TokenTracker

log = get_logger("memory")


class MemoryManager:
    """检索记忆管理器：存储 + token 估算 + 超限压缩。"""

    def __init__(
        self,
        make_item: Callable[..., Any],
        token_tracker: Optional[TokenTracker] = None,
        max_tokens: Optional[int] = None,
        keep_recent: Optional[int] = None,
        model: str = "memory",
    ) -> None:
        """
        :param make_item: 构造摘要条目的工厂（如 KnowledgeItem），按关键字 question/answer/type 调用
        :param token_tracker: 复用主流程的 token 统计器，压缩消耗也计入总预算
        :param max_tokens: 记忆 token 上限，超过则触发压缩（默认读环境变量 MEMORY_MAX_TOKENS）
        :param keep_recent: 压缩时保留的最近条目数（默认读环境变量 MEMORY_KEEP_RECENT）
        :param model: 压缩所用模型名（经 get_model 解析，缺省回退到默认工具配置）
        """
        self.items: List[Any] = []
        self._make_item = make_item
        self.token_tracker = token_tracker
        self.max_tokens = int(max_tokens if max_tokens is not None else os.getenv("MEMORY_MAX_TOKENS", "30000"))
        self.keep_recent = int(keep_recent if keep_recent is not None else os.getenv("MEMORY_KEEP_RECENT", "12"))
        self.model = model

    # ---------------- 基本容器接口 ----------------
    def add(self, item: Any) -> None:
        self.items.append(item)

    def extend(self, items: Any) -> None:
        for it in items:
            self.items.append(it)

    def get_items(self) -> List[Any]:
        """读取当前全部记忆（供每轮迭代注入 prompt）。"""
        return self.items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    # ---------------- token 估算 ----------------
    @staticmethod
    def _item_text(item: Any) -> str:
        q = getattr(item, "question", "") or ""
        a = getattr(item, "answer", "") or ""
        return f"Q: {q}\nA: {a}"

    def estimate_tokens(self, items: Optional[List[Any]] = None) -> int:
        items = self.items if items is None else items
        if not items:
            return 0
        text = "\n\n".join(self._item_text(it) for it in items)
        try:
            return count_text_tokens([{"role": "user", "content": text}])
        except Exception:
            # tokenizer 不可用时退化为按字符粗估（约 1 token / 2 字符）
            return len(text) // 2

    # ---------------- 压缩 / 整理 ----------------
    async def consolidate(self, reason: str = "iteration") -> bool:
        """读取并整理记忆：若超出 token 上限，则压缩较早的记忆。

        :return: 是否发生了压缩
        """
        if not self.items:
            return False

        total = self.estimate_tokens()
        if total <= self.max_tokens:
            return False

        log.info(
            f"[memory] 记忆超出上限({total} > {self.max_tokens} tokens, reason={reason})，开始压缩 "
            f"(共 {len(self.items)} 条, 保留最近 {self.keep_recent} 条)"
        )

        if self.keep_recent > 0:
            recent = self.items[-self.keep_recent:]
            older = self.items[:-self.keep_recent]
        else:
            recent = []
            older = list(self.items)

        if not older:
            # 仅最近条目就已超限，无法继续压缩（避免丢失最新信息）
            log.warning("[memory] 最近条目已超过上限，跳过压缩")
            return False

        summary_item = await self._summarize(older)
        new_items: List[Any] = []
        if summary_item is not None:
            new_items.append(summary_item)
        new_items.extend(recent)

        # 原地替换，保证外部引用同一列表者同步可见
        self.items[:] = new_items
        log.info(
            f"[memory] 压缩完成：{len(older)} 条 -> 1 条摘要，"
            f"当前 {len(self.items)} 条 / {self.estimate_tokens()} tokens"
        )
        return True

    async def _summarize(self, items: List[Any]) -> Optional[Any]:
        """调用 LLM 把一批记忆压缩成一条摘要条目。"""
        joined = "\n\n".join(
            f"[{i + 1}] 问题: {getattr(it, 'question', '')}\n回答: {getattr(it, 'answer', '')}"
            for i, it in enumerate(items)
        )
        # 防止摘要输入过长
        joined = joined[:24000]

        system = (
            "你是一个研究记忆压缩器。你的任务是把多条检索得到的知识压缩为一份简洁、"
            "结构化、信息无损的摘要，保留关键事实、数值、结论、实体与来源线索，删除重复与冗余表述。"
            "只输出摘要正文，不要添加解释或客套话。"
        )
        prompt = (
            "请将以下检索记忆压缩为一份简洁但信息完整的中文摘要"
            "（可用要点列表组织，确保后续推理所需的关键信息不丢失）：\n\n"
            f"{joined}"
        )

        text = ""
        try:
            res = ai_generate_object(
                model=self.model,
                schema=None,
                system=system,
                prompt=prompt,
                maxTokens=16384,
                temperature=0.2,
            )
            if self.token_tracker is not None:
                try:
                    self.token_tracker.track_usage(self.model, res.get("usage", {}))
                except Exception:
                    pass
            text = res.get("object") or ""
            if isinstance(text, dict):
                text = json.dumps(text, ensure_ascii=False)
        except Exception as e:
            log.warning(f"[memory] 摘要生成失败，退化为截断保留：{e}")
            text = joined[:4000]

        try:
            return self._make_item(
                question="历史检索记忆摘要（已压缩）",
                answer=text,
                type="memory",
            )
        except Exception as e:
            log.error(f"[memory] 构造摘要条目失败：{e}")
            return None
