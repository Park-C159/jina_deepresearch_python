"""
插件管理器。
- 自动发现 plugins/ 目录下的模块
- 延迟实例化并传入配置
- 提供统一的 get_plugin_instance 接口
"""
import importlib
import os
import pkgutil
from pathlib import Path
from typing import Dict, Any, Optional

from core.plugin_registry import _REGISTRY, get_plugin, list_plugins, has_plugin
from core.plugin_protocols import BasePlugin

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLUGINS_PACKAGE = "plugins"


class PluginManager:
    """插件管理器：负责发现、加载、缓存插件实例。"""

    def __init__(self, plugins_package: str = _PLUGINS_PACKAGE):
        self.plugins_package = plugins_package
        self._instances: Dict[str, Dict[str, BasePlugin]] = {}

    def discover(self) -> None:
        """自动扫描并导入 plugins 包下的所有子模块，触发装饰器注册。"""
        try:
            importlib.import_module(self.plugins_package)
        except ImportError:
            return

        package = importlib.import_module(self.plugins_package)
        package_path = getattr(package, "__path__", [])

        def _walk_package(pkg_path: list, prefix: str):
            for _, modname, ispkg in pkgutil.iter_modules(pkg_path, prefix):
                try:
                    importlib.import_module(modname)
                except Exception as e:
                    print(f"[PluginManager] Failed to import {modname}: {e}")
                    continue
                if ispkg:
                    try:
                        subpkg = importlib.import_module(modname)
                        sub_path = getattr(subpkg, "__path__", [])
                        _walk_package(sub_path, modname + ".")
                    except Exception as e:
                        print(f"[PluginManager] Failed to walk subpackage {modname}: {e}")

        _walk_package(package_path, self.plugins_package + ".")

    def get_instance(
        self,
        category: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BasePlugin:
        """
        获取（或创建）插件实例。
        实例会被缓存，同一个 category/name 只初始化一次。
        """
        if category not in self._instances:
            self._instances[category] = {}
        if name not in self._instances[category]:
            plugin_cls = get_plugin(category, name)
            instance = plugin_cls()
            if config is not None:
                instance.initialize(config)
            self._instances[category][name] = instance
        return self._instances[category][name]

    def reload(self, category: str, name: str) -> BasePlugin:
        """重新加载某个插件实例（配置热更新时用）"""
        if category in self._instances and name in self._instances[category]:
            del self._instances[category][name]
        return self.get_instance(category, name)

    def list(self, category: str = None) -> Dict[str, Any]:
        return list_plugins(category)


# -------------- 全局便捷函数 --------------

_manager: Optional[PluginManager] = None


def get_manager() -> PluginManager:
    """获取全局单例 PluginManager，首次调用会自动 discover。"""
    global _manager
    if _manager is None:
        _manager = PluginManager()
        _manager.discover()
    return _manager


def get_plugin_instance(
    category: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BasePlugin:
    """快捷函数：获取插件实例"""
    return get_manager().get_instance(category, name, config)


def discover_plugins() -> None:
    """手动触发一次插件发现"""
    get_manager().discover()
