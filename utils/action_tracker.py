# action_tracker.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, DefaultDict, Dict, List, Optional
from copy import deepcopy
from collections import defaultdict


# ---- 轻量 EventEmitter（与之前 TokenTracker 里的实现一致） ----
class _EventEmitter:
    def __init__(self) -> None:
        self._listeners: DefaultDict[str, List[Callable[..., None]]] = defaultdict(list)

    def on(self, event: str, fn: Callable[..., None]) -> None:
        self._listeners[event].append(fn)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for fn in list(self._listeners.get(event, [])):
            try:
                fn(*args, **kwargs)
            except Exception:
                # 不让单个监听器异常影响其他监听器
                import logging
                logging.getLogger(__name__).exception("Error in '%s' listener", event)


# ---- 你自己的 i18n 文本函数；用你项目里的实现替换 ----
def get_i18n_text(text: str, lang: Optional[str] = None, params: Dict[str, Any] | None = None) -> str:
    # 占位：直接返回；你可接入实际的 i18n 实现
    return text


# ---- 类型定义 ----
@dataclass
class StepAction:
    action: str = "answer"
    answer: str = ""
    references: List[str] = field(default_factory=list)
    think: str = ""
    URLTargets: List[str] = field(default_factory=list)


@dataclass
class ActionState:
    thisStep: StepAction = field(default_factory=StepAction)
    gaps: List[str] = field(default_factory=list)
    totalStep: int = 0


# ---- 主类 ----
class ActionTracker(_EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self._state: ActionState = ActionState()

    # 等价于 TS: trackAction(newState: Partial<ActionState>)
    def track_action(self, new_state: Dict[str, Any]) -> None:
        """
        接收一个部分字段的新状态并合并（浅合并 + thisStep 的字段级合并）。
        例如：
        track_action({"thisStep": {"answer": "ok"}, "totalStep": 3})
        """
        current = self._state

        # 合并 thisStep
        if "thisStep" in new_state and isinstance(new_state["thisStep"], dict):
            step_dict = asdict(current.thisStep)
            step_dict.update(new_state["thisStep"])  # 浅合并字段
            current = replace(current, thisStep=StepAction(**step_dict))

        # 合并 gaps
        if "gaps" in new_state and isinstance(new_state["gaps"], list):
            current = replace(current, gaps=list(new_state["gaps"]))

        # 合并 totalStep
        if "totalStep" in new_state and isinstance(new_state["totalStep"], int):
            current = replace(current, totalStep=new_state["totalStep"])

        self._state = current
        # 触发 'action' 事件，载荷与 TS 一致：当前 thisStep
        self.emit("action", deepcopy(self._state.thisStep))

    # 等价于 TS: trackThink(think: string, lang?: string, params = {})
    def track_think(self, think: str, lang: Optional[str] = None, params: Dict[str, Any] | None = None) -> None:
        if lang:
            think = get_i18n_text(think, lang, params or {})
        # 更新 thisStep.think，且清空 URLTargets（与 TS 行为一致）
        updated_step = replace(self._state.thisStep, think=think, URLTargets=[])
        self._state = replace(self._state, thisStep=updated_step)
        self.emit("action", deepcopy(self._state.thisStep))

    # 等价于 TS: getState()
    def get_state(self) -> ActionState:
        # 返回一个拷贝，避免外部修改内部状态
        return deepcopy(self._state)

    # 等价于 TS: reset()
    def reset(self) -> None:
        self._state = ActionState()


# ---- 使用示例 ----
if __name__ == "__main__":
    tracker = ActionTracker()

    # 订阅 'action' 事件
    tracker.on("action", lambda step: print("[EVENT action]", step))

    tracker.track_action({"thisStep": {"action": "answer", "answer": "Hello"}, "totalStep": 1})
    tracker.track_think("先想一想", lang="zh", params={"name": "demo"})
    print("STATE:", tracker.get_state())
    tracker.reset()
    print("AFTER RESET:", tracker.get_state())
