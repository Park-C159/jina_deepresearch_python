"""DeepStatic 插件核心基础设施"""
from core.plugin_registry import register_plugin, get_plugin, list_plugins, has_plugin
from core.plugin_manager import PluginManager, discover_plugins, get_plugin_instance
from core.plugin_protocols import BasePlugin, BaseLLMPlugin, BaseSearchPlugin, BaseToolPlugin

__all__ = [
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "has_plugin",
    "PluginManager",
    "discover_plugins",
    "get_plugin_instance",
    "BasePlugin",
    "BaseLLMPlugin",
    "BaseSearchPlugin",
    "BaseToolPlugin",
]
