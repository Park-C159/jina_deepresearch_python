import os
from typing import TypedDict, Literal, Optional, Dict, Any
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from google.cloud import aiplatform

# 加载.env文件到环境变量
load_dotenv()

from config.config_loader import config

# 临时provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER")
GCLOUD_PROJECT = os.getenv("GCLOUD_PROJECT")
STEP_SLEEP = config['defaults']['step_sleep']

_OPENAI_VALID_KEYS = {
    "api_key", "base_url", "timeout", "max_retries", "default_headers",
    "default_query", "http_client", "_strict_response", "organization", "project"
}


class ToolConfig(TypedDict):
    model: str
    temperature: float
    maxTokens: int


class ToolOverrides(TypedDict, total=False):
    model: Optional[str]
    temperature: Optional[float]
    maxTokens: Optional[int]


def get_tool_config(tool_name) -> ToolConfig:
    """返回指定工具的完整配置（合并 default + tool 级覆盖）"""
    provider_key = "gemini" if LLM_PROVIDER == "vertex" else LLM_PROVIDER
    provider_config = config["models"][provider_key]

    default: ToolConfig = provider_config["default"]  # type: ignore

    overrides: ToolOverrides = provider_config["tools"].get(tool_name, {})  # type: ignore

    res = ToolConfig(
        model=overrides.get("model") or default["model"],
        temperature=overrides.get("temperature") or default["temperature"],
        maxTokens=overrides.get("maxTokens") or default["maxTokens"],
    )
    return res


# --------------- 创建模型实例 ---------------
def get_model(tool_name: str):
    """返回 (已配置好的客户端, compatibility, model_name)"""
    cfg = get_tool_config(tool_name)  # 里边应包含 cfg["model"] = "gpt-4o-mini" 之类
    provider_cfg: Dict[str, Any] = config["providers"].get(LLM_PROVIDER, {})

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not found")

        raw_cfg = provider_cfg.get("clientConfig") or {}
        client_kwargs = {"api_key": OPENAI_API_KEY}
        client_kwargs.update({k: v for k, v in raw_cfg.items() if k in _OPENAI_VALID_KEYS})

        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL

        compatibility = raw_cfg.get("compatibility")  # 可能为 None
        return openai.OpenAI(**client_kwargs), compatibility, cfg["model"]

    if LLM_PROVIDER == "vertex":
        if not GCLOUD_PROJECT:
            raise RuntimeError("GCLOUD_PROJECT not found")
        aiplatform.init(project=GCLOUD_PROJECT, **(provider_cfg.get("clientConfig") or {}))
        return aiplatform.ChatModel.from_pretrained(cfg["model"]), '', cfg["model"]

    # 默认 Gemini
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not found")
    genai.configure(api_key=GEMINI_API_KEY, **(provider_cfg.get("clientConfig") or {}))
    return genai.GenerativeModel(cfg["model"]), '', cfg["model"]


def get_client(model: str) -> openai.OpenAI:
    if LLM_PROVIDER == "qwen":
        return openai.OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if LLM_PROVIDER == "openai":
        return openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
    # 继续扩展 vertex / gemini ...
    raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")
