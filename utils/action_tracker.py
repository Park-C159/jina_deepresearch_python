# action_tracker.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, DefaultDict, Dict, List, Optional
from copy import deepcopy
from collections import defaultdict
import threading

from tool.text_tools import get_i18n_text


class EventEmitter:
    def __init__(self) -> None:
        self._events: Dict[str, List[Callable[..., Any]]] = {}
        self._lock = threading.RLock()
        self._max_listeners: Optional[int] = None  # 跟 Node 一样，默认不限制

    # 别名：add_listener == on
    def on(self, event: str, listener: Callable[..., Any]) -> "EventEmitter":
        with self._lock:
            listeners = self._events.setdefault(event, [])
            if self._max_listeners is not None and len(listeners) >= self._max_listeners:
                raise RuntimeError(f"Max listeners exceeded for event '{event}'")
            listeners.append(listener)
        return self

    add_listener = on

    def once(self, event: str, listener: Callable[..., Any]) -> "EventEmitter":
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            # 调用后移除
            self.off(event, _wrapper)
            return listener(*args, **kwargs)

        # 便于 off 时按原始函数卸载
        setattr(_wrapper, "__original_listener__", listener)
        return self.on(event, _wrapper)

    # 别名：remove_listener == off
    def off(self, event: str, listener: Callable[..., Any]) -> "EventEmitter":
        with self._lock:
            if event not in self._events:
                return self
            wrapped = self._events[event]

            # 既匹配 wrapper 也匹配其原 listener
            def _is_match(fn: Callable[..., Any]) -> bool:
                original = getattr(fn, "__original_listener__", None)
                return fn is listener or original is listener

            self._events[event] = [fn for fn in wrapped if not _is_match(fn)]
            if not self._events[event]:
                self._events.pop(event, None)
        return self

    remove_listener = off

    def remove_all_listeners(self, event: Optional[str] = None) -> "EventEmitter":
        with self._lock:
            if event is None:
                self._events.clear()
            else:
                self._events.pop(event, None)
        return self

    def emit(self, event: str, *args: Any, **kwargs: Any) -> bool:
        with self._lock:
            listeners = list(self._events.get(event, []))
        # 在锁外调用，避免监听函数里再次发射或卸载导致死锁
        for fn in listeners:
            fn(*args, **kwargs)
        return len(listeners) > 0

    def listeners(self, event: str) -> List[Callable[..., Any]]:
        with self._lock:
            return list(self._events.get(event, []))

    def listener_count(self, event: str) -> int:
        with self._lock:
            return len(self._events.get(event, []))

    def set_max_listeners(self, n: Optional[int]) -> "EventEmitter":
        if n is not None and n < 0:
            raise ValueError("max listeners must be >= 0 or None")
        self._max_listeners = n
        return self


@dataclass
class StepAction:
    def __init__(
            self,
            action: str = "answer",
            answer: str = "",
            references: List[str] | None = None,
            think: str = "",
            URL_target: List[str] | None = None,
            search_requests: List[str] | None = None,
            question2answer: str | None = None,
            coding_issue: str | None = None,
            isFinal: bool = True
    ):
        self.action = action
        self.answer = answer
        self.references = references or []
        self.think = think
        self.URL_target = URL_target or []
        self.search_requests = search_requests or []
        self.question2answer = question2answer
        self.coding_issue = coding_issue
        self.isFinal = isFinal

    def copy(self) -> StepAction:
        return StepAction(
            action=self.action,
            answer=self.answer,
            references=self.references.copy(),
            think=self.think,
            URL_target=self.URL_target.copy(),
            search_requests=self.search_requests.copy(),
            question2answer=self.question2answer,
            coding_issue=self.coding_issue,
            isFinal=self.isFinal,
        )


# -------------------------------------------------
class ActionState:
    __slots__ = ("thisStep", "gaps", "totalStep")

    def __init__(
            self,
            thisStep: StepAction | None = None,
            gaps: List[str] | None = None,
            totalStep: int = 0,
    ):
        self.thisStep = thisStep or StepAction()
        self.gaps = gaps or []
        self.totalStep = totalStep

    def copy(self) -> ActionState:
        return ActionState(
            thisStep=self.thisStep.copy(),
            gaps=self.gaps.copy(),
            totalStep=self.totalStep,
        )


# -------------------------------------------------
class ActionTracker(EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self._state = ActionState()

    # -------------------- 核心 API --------------------
    def track_action(self, new_state: Dict[str, Any]) -> None:
        for key, value in new_state.items():
            if key == "thisStep" and isinstance(value, dict):
                # 把字典转成 StepAction
                value = StepAction(**value)
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self.emit("action", self._state.thisStep)

    def track_think(
            self,
            think: str,
            lang: str | None = None,
            params: Dict[str, Any] | None = None,
    ) -> None:
        if lang is not None:
            think = get_i18n_text(think, lang, params)

        # 更新 think 并清空 URLTargets（与 TS 侧一致）
        new_step = self._state.thisStep.copy()
        new_step.think = think
        new_step.URL_target = []

        self._state.thisStep = new_step
        self.emit("action", self._state.thisStep)

    def get_state(self) -> ActionState:
        """返回当前状态的深拷贝，避免外部篡改"""
        return self._state.copy()

    def reset(self) -> None:
        """重置为初始状态"""
        self._state = ActionState()
        # 如需通知监听者，可再 emit


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
