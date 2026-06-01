"""Google Cloud Vertex AI LLM 插件"""
import os
from typing import Any, Dict

try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None

from core.plugin_protocols import BaseLLMPlugin
from core.plugin_registry import register_plugin
from config.config_loader import config


@register_plugin("llm", "vertex")
class VertexLLMPlugin(BaseLLMPlugin):
    """Google Cloud Vertex AI 模型插件"""

    def initialize(self, cfg: Dict[str, Any]) -> None:
        if aiplatform is None:
            raise RuntimeError("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        self.project = cfg.get("project") or os.getenv("GCLOUD_PROJECT")
        self.provider_cfg = config["providers"].get("gemini", {})
        self.models_cfg = config["models"].get("gemini", {})
        if not self.project:
            raise RuntimeError("GCLOUD_PROJECT not found")
        aiplatform.init(project=self.project, **(self.provider_cfg.get("clientConfig") or {}))

    def health_check(self) -> bool:
        try:
            # Vertex 健康检查较复杂，简化处理
            return bool(self.project)
        except Exception:
            return False

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        default = self.models_cfg.get("default", {})
        overrides = self.models_cfg.get("tools", {}).get(tool_name, {})
        return {
            "model": overrides.get("model") or default.get("model", "gemini-2.5-flash"),
            "temperature": overrides.get("temperature") if overrides.get("temperature") is not None else default.get("temperature", 0),
            "maxTokens": overrides.get("maxTokens") or default.get("maxTokens", 8192),
        }

    def get_client(self, tool_name: str) -> tuple:
        cfg = self.get_tool_config(tool_name)
        return aiplatform.ChatModel.from_pretrained(cfg["model"]), "", cfg["model"]

    def get_raw_client(self) -> Any:
        return aiplatform
