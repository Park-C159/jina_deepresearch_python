from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, DefaultDict
from collections import defaultdict
import logging
import contextvars

from utils.action_tracker import EventEmitter

logger = logging.getLogger("TokenTracker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------
# 可选：用 contextvars 模拟“异步上下文中的收费值”
# 在你的程序里随处可读 charge_amount_ctx.get(None)
# ---------------------------
charge_amount_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "charge_amount", default=None
)


@dataclass
class LanguageModelUsage:
    promptTokens: int = 0
    completionTokens: int = 0
    totalTokens: int = 0
    # 可选字段（如果你有的话，可自行补充）
    reasoningTokens: Optional[int] = None
    cachedInputTokens: Optional[int] = None


@dataclass
class TokenUsage:
    tool: str
    usage: LanguageModelUsage


class TokenTracker(EventEmitter):
    def __init__(self, budget: Optional[int] = None) -> None:
        super().__init__()
        self.usages = []
        self.budget: Optional[int] = budget

        # 模拟 Node 里的 asyncLocalContext 钩子：
        # 监听 'usage' 事件，更新 contextvar
        def _on_usage(_usage: LanguageModelUsage) -> None:
            # 这里与原逻辑一致：把总 Token 写进“异步上下文”
            total = self.get_total_usage().totalTokens
            try:
                charge_amount_ctx.set(total)
            except Exception:
                # 如果使用者不需要该功能，忽略即可
                pass

        self.on("usage", _on_usage)

    # 对应 TS: trackUsage(tool, usage)
    def track_usage(self, tool: str, usage) -> None:
        self.usages.append({
            "tool": tool,
            "usage": usage,
        })
        self.emit("usage", usage)

    # 对应 TS: getTotalUsage(): LanguageModelUsage
    def get_total_usage(self) -> LanguageModelUsage:
        # scaler 目前恒为 1，保持与 TS 版本一致
        total = LanguageModelUsage()
        for item in self.usages:
            u = item.get("usage")
            prompt_tokens = u["promptTokens"]
            completion_tokens = u["completionTokens"]
            total_tokens = u["totalTokens"]
            scaler = 1
            total.promptTokens += (prompt_tokens or 0) * scaler
            total.completionTokens += (completion_tokens or 0) * scaler
            total.totalTokens += (total_tokens or 0) * scaler
        print(total)
        return total

    # 对应 TS: getTotalUsageSnakeCase()
    def get_total_usage_snake_case(self) -> Dict[str, int]:
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for item in self.usages:
            u = item.get("usage")
            scaler = 1
            acc["prompt_tokens"] += (u.get("promptTokens") or 0) * scaler
            acc["completion_tokens"] += (u.get("completionTokens") or 0) * scaler
            acc["total_tokens"] += (u.get("totalTokens") or 0) * scaler
        return acc

    # 对应 TS: getUsageBreakdown(): Record<string, number>
    def get_usage_breakdown(self) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for item in self.usages:
            breakdown[item.tool] = breakdown.get(item.tool, 0) + (item.usage.totalTokens or 0)
        return breakdown

    # 对应 TS: printSummary()
    def print_summary(self) -> None:
        breakdown = self.get_usage_breakdown()
        total = self.get_total_usage()
        logger.info(
            "Token Usage Summary | budget=%s | total=%s | breakdown=%s",
            self.budget,
            {
                "promptTokens": total.promptTokens,
                "completionTokens": total.completionTokens,
                "totalTokens": total.totalTokens,
            },
            breakdown,
        )

    # 对应 TS: reset()
    def reset(self) -> None:
        self.usages.clear()
