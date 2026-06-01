"""阿里云通义千问（DashScope）LLM 插件"""
import os
from typing import Any, Dict

import openai

from core.plugin_protocols import BaseLLMPlugin
from core.plugin_registry import register_plugin
from config.config_loader import config


@register_plugin("llm", "qwen")
class QwenLLMPlugin(BaseLLMPlugin):
    """通义千问 DashScope 插件。底层协议兼容 OpenAI。"""

    def initialize(self, cfg: Dict[str, Any]) -> None:
        self.api_key = cfg.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = cfg.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.models_cfg = config["models"].get("qwen", config["models"].get("openai", {}))
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not found")

    def health_check(self) -> bool:
        try:
            client = self.get_raw_client()
            client.models.list()
            return True
        except Exception:
            return False

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        default = self.models_cfg.get("default", {})
        overrides = self.models_cfg.get("tools", {}).get(tool_name, {})
        return {
            "model": overrides.get("model") or default.get("model", "qwen-turbo"),
            "temperature": overrides.get("temperature") if overrides.get("temperature") is not None else default.get("temperature", 0),
            "maxTokens": overrides.get("maxTokens") or default.get("maxTokens", 8192),
        }

    def get_client(self, tool_name: str) -> tuple:
        cfg = self.get_tool_config(tool_name)
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return client, "tools", cfg["model"]

    def get_raw_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
