# DeepStatic 插件接入指南

## 架构概览

```
core/
  plugin_protocols.py   # 基类 / 协议定义
  plugin_registry.py    # @register_plugin 装饰器注册表
  plugin_manager.py     # 自动发现 + 实例化
plugins/
  llm/                  # LLM 模型插件
  search/               # 搜索插件
  tools/                # 自定义工具插件
```

## 三步接入新工具

### 第 1 步：选择基类

| 你想扩展的能力 | 继承的基类 | 存放目录 |
|---|---|---|
| 大语言模型（LLM） | `BaseLLMPlugin` | `plugins/llm/` |
| 搜索引擎 | `BaseSearchPlugin` | `plugins/search/` |
| 任意自定义工具 | `BaseToolPlugin` | `plugins/tools/` |

### 第 2 步：实现插件类

以接入 **Brave Search** 为例：

```python
# plugins/search/brave_search.py
import os
from typing import Any, Dict, Optional
import aiohttp

from core.plugin_protocols import BaseSearchPlugin
from core.plugin_registry import register_plugin
from utils.token_tracker import TokenTracker


@register_plugin("search", "brave")
class BraveSearchPlugin(BaseSearchPlugin):
    def initialize(self, cfg: Dict[str, Any]) -> None:
        self.api_key = cfg.get("api_key") or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise RuntimeError("BRAVE_API_KEY not found")

    def health_check(self) -> bool:
        return bool(self.api_key)

    async def search(self, query, domain=None, num=None, meta=None, tracker=None):
        q = query.get("q", "")
        headers = {"X-Subscription-Token": self.api_key}
        params = {"q": q, "count": num or 20}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers, params=params,
            ) as resp:
                data = await resp.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
            })

        return {"response": {"results": results}}
```

**关键点**：
- 必须用 `@register_plugin("search", "brave")` 装饰器注册
- `initialize(cfg)` 接收配置字典（来自 `settings.json` 或环境变量）
- 搜索插件返回统一格式 `{"response": {"results": [...]}}`

### 第 3 步：配置并运行

1. **环境变量**：
   ```bash
   export SEARCH_PROVIDER=brave
   export BRAVE_API_KEY=your_key_here
   ```

2. **可选：settings.json 增加配置**（如需自定义参数）：
   ```json
   {
     "providers": {
       "brave": {
         "api_key": "your_key",
         "base_url": "https://api.search.brave.com/res/v1/web/search"
       }
     }
   }
   ```

3. **重启服务**：
   ```bash
   python web_ui.py
   ```

---

## 接入新 LLM 模型示例

```python
# plugins/llm/my_custom_llm.py
import openai
from core.plugin_protocols import BaseLLMPlugin
from core.plugin_registry import register_plugin

@register_plugin("llm", "myllm")
class MyCustomLLMPlugin(BaseLLMPlugin):
    def initialize(self, cfg):
        self.api_key = cfg.get("api_key") or os.getenv("MYLLM_API_KEY")
        self.base_url = cfg.get("base_url")

    def health_check(self):
        return bool(self.api_key)

    def get_client(self, tool_name):
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return client, "tools", "my-model"

    def get_raw_client(self):
        return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_tool_config(self, tool_name):
        return {"model": "my-model", "temperature": 0, "maxTokens": 8192}
```

然后设置 `LLM_PROVIDER=myllm`。

---

## 运行方式

### 命令行
```bash
# 使用插件化的搜索和模型
export SEARCH_PROVIDER=jina      # 或 milvus / brave
export LLM_PROVIDER=openai       # 或 gemini / qwen / vertex
python agent.py --question "你的问题" --search_provider jina
```

### Web UI
```bash
python web_ui.py
```
Web 界面会自动读取已注册的插件列表，下拉框中可直接选择任意已注册的搜索提供商和 LLM 提供商。

### 查看已注册插件
```python
from core.plugin_manager import get_manager
mgr = get_manager()
print(mgr.list())  # {'llm': {...}, 'search': {...}}
```
