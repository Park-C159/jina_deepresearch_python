"""
插件注册表。
使用装饰器注册，运行时通过名称查找。
"""
from typing import Dict, Type, Any

_REGISTRY: Dict[str, Dict[str, Type]] = {}


def register_plugin(category: str, name: str):
    """
    装饰器：将类注册为指定分类和名称的插件。

    用法示例:
        @register_plugin("search", "jina")
        class JinaSearchPlugin(BaseSearchPlugin):
            ...
    """

    def decorator(cls):
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        if name in _REGISTRY[category]:
            raise ValueError(
                f"Plugin '{category}/{name}' already registered "
                f"by {_REGISTRY[category][name].__module__}.{ _REGISTRY[category][name].__name__}"
            )
        _REGISTRY[category][name] = cls
        cls.name = name
        cls.category = category
        return cls

    return decorator


def get_plugin(category: str, name: str) -> Type:
    """按分类和名称获取插件类"""
    if category not in _REGISTRY:
        available_cats = list(_REGISTRY.keys())
        raise ValueError(
            f"Plugin category '{category}' not found. Available categories: {available_cats}"
        )
    if name not in _REGISTRY[category]:
        available = list(_REGISTRY[category].keys())
        raise ValueError(
            f"Plugin '{category}/{name}' not found. Available in '{category}': {available}"
        )
    return _REGISTRY[category][name]


def list_plugins(category: str = None) -> Dict[str, Any]:
    """列出已注册插件。不指定 category 则返回全部。"""
    if category:
        return dict(_REGISTRY.get(category, {}))
    return {k: dict(v) for k, v in _REGISTRY.items()}


def has_plugin(category: str, name: str) -> bool:
    """检查插件是否已注册"""
    return category in _REGISTRY and name in _REGISTRY[category]


def unregister_plugin(category: str, name: str) -> bool:
    """注销插件（主要用于测试）"""
    if has_plugin(category, name):
        del _REGISTRY[category][name]
        return True
    return False
