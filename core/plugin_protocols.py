"""
插件协议/基类定义。
所有插件必须继承对应的基类，并实现抽象方法。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BasePlugin(ABC):
    """所有插件的公共基类"""

    name: str = ""
    category: str = ""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化插件。
        :param config: 从 settings.json / 环境变量解析出的配置字典
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查，返回是否可用"""
        ...


class BaseLLMPlugin(BasePlugin):
    """
    LLM 模型插件协议。
    负责：创建客户端、提供模型配置、执行生成。
    """

    @abstractmethod
    def get_client(self, tool_name: str) -> tuple:
        """
        返回三元组: (client, compatibility, model_name)
        - client: 可直接用于 instructor 或原生调用的客户端对象
        - compatibility: "tools" | "json" | "json_schema" | "strict" | None
        - model_name: 实际请求的模型 ID 字符串
        """
        ...

    @abstractmethod
    def get_raw_client(self) -> Any:
        """返回最底层的 SDK 客户端（如 openai.OpenAI）"""
        ...

    @abstractmethod
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """返回指定工具的合并配置 {model, temperature, maxTokens}"""
        ...


class BaseSearchPlugin(BasePlugin):
    """
    搜索插件协议。
    负责：接收查询字典，返回搜索结果。
    """

    @abstractmethod
    async def search(
        self,
        query: Dict[str, Any],
        domain: Optional[str] = None,
        num: Optional[int] = None,
        meta: Optional[str] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        执行搜索。
        :param query: 搜索参数字典，至少包含 {"q": "搜索关键词"}
        :param domain: 限定搜索域（如 "arxiv"）
        :param num: 返回结果数量
        :param meta: 元信息
        :param tracker: TokenTracker 实例（可选）
        :return: 统一返回格式 {"response": {...}}，内部必须包含可解析的结果列表
        """
        ...


class BaseToolPlugin(BasePlugin):
    """
    通用工具插件协议。
    用于扩展任意自定义工具能力。
    """

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具逻辑"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，供 Agent 决策时参考"""
        ...
