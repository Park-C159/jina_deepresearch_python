# 这是一个示例 Python 脚本。
import asyncio
import re, sys
from typing import List, Dict, Any, Optional
from pydantic import create_model, BaseModel, Field

from tool.serp_cluster import serp_cluster
from utils.action_tracker import ActionTracker
from utils.get_log import get_logger
from utils.url_tool import add_to_all_urls

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

messages = [
    {'role': 'system', 'content': '你是一位历史研究专家。'},
    {'role': 'user', 'content': '三大战役各个军队伤亡如何？'},
    {
        'role': 'assistant',
        'content': '<think>三大战役是指解放战争（1946–1949）期间，中国民解放军与国民党军进行的三次战略性决战：辽沈战役、淮海战役和平津战役。这三场战役决定了中国内战的走向，对国共双方的兵力和士气造成了重大影响。以下是各战役中双方军队伤亡情况的大致统计（数据主要依据官方   资料和历史研究，存在一定的估算成分）：\n\n---\n\n### 一、辽沈战役（1948年9月12日－11月2日）\n\n**地点**：东北地区（主要在辽宁、吉林）\n\n**参战方**：\n- 中国人民解放军：东北野战军（林彪、罗荣桓指挥）\n- 国民党军：东北“剿总”部队（卫立煌等指挥）\n\n**兵力对比**：\n- 解放军：约70万人\n- 国民党军：约55万人\n\n**伤亡与损失**：\n- **解放军伤亡**：约6.9万人（其中阵亡约1.4万人）\n- **国民党军伤亡及被俘**： 约47.2万人  \n  （包括阵亡、受伤、被俘、起义或投诚）\n\n**结果**：解放军全歼国民党在东北的主力，解放整个东北。\n\n---\n\n### 二、淮海战役（1948年11月6日－1949年1月10日）\n\n**地点**：以徐州为中心的华东中原地区（江苏、安徽、河南、山东交界）\n\n**参战方**：\n- 中国人民解放军：华东野战军（粟裕、陈毅）、中原野战军（刘伯承、邓小平）\n- 国民党军：徐州“剿总”部队（刘峙、杜聿明等指挥）\n\n**兵力对比**：\n- - 解放军：约60万人\n- 国民党军：约80万人（战役期间陆续投入）\n\n**伤亡与损失**：\n- **解放军伤亡**：约13.6万人（部分资料称13.4万）\n- **国民党军损失**：约55.5万人  \n  （其中被歼灭、俘虏、起义或投诚，包黄百韬兵团、黄维兵团、杜聿明集团等主力）\n\n**特点**：解放军以少胜多，是三大战役中歼敌最多、规模最大、最艰苦的一战。\n\n---\n\n### 三、平津战役（1948年11月29日－1949年1月31日）\n\n**地点**：华北地区   （北平、天津、张家口等地）\n\n**参战方**：\n- 中国人民解放军：东北野战军、华北野战军（林彪、罗荣桓、聂荣臻指挥）\n- 国民党军：傅作义部（华北“剿总”）\n\n**兵力对比**：\n- 解放军：约100万人（含东北野战军关后兵力）\n- 国民党军：约60万人\n\n**伤亡与损失**：\n- **解放军伤亡**：约3.9万人\n- **国民党军损失**：约52万人  \n  - 天津战斗中歼灭约13万人（陈长捷部被全歼）\n  - 北平和平解放，傅作义率25万余人接受受编\n  - 张家口、新保安等地歼敌约10余万\n\n**结果**：北平和平解放，避免了古都毁于战火，华北基本解放。\n\n---\n\n### 总结：三大战役整体伤亡情况\n\n| 战役     | 解放军伤亡 | 国民党军损失 | 总歼敌数 |\n| |----------|------------|--------------|----------|\n| 辽沈战役 | 约6.9万    | 约47.2万     | 47.2万   |\n| 淮海战役 | 约13.6万   | 约55.5万     | 55.5万   |\n| 平津战役 | 约3.9万    | 约52万       | 52万     |\n| **总计** | **约24.4万** | **约154.7万** | **154.7万** |\n\n---\n\n### 补充说明：\n\n1. **数据来源**：以上数字主要来自《中国人民解放军战史》《毛泽东军事文集》及中共中央党史研究室出版的权威资。\n2. **“损失”含义**：国民党军的“损失”不仅包括阵亡和负伤，还包括被俘、起义、投诚和改编人员，因此总数远高于实际战斗死亡人数。\n3. **战略意义**：三大战役后，国民党主力基本被消灭，解放军掌握了战略主动   权，为渡江战役和全国解放奠定了基础。\n\n如需更详细的部队番号、将领信息或战役过程分析，也可进一步提供。</think>\n三大战役是指解放战争（1946–1949）期间，中国人民解放军与国民党军进行的三次战略性决战：辽沈役、淮海战役和平津战役。这三场战役决定了中国内战的走向，对国共双方的兵力和士气造成了重大影响。以下是各战役中双方军队伤亡情况的大致统计（数据主要依据官方资料和历史研究，存在一定的估算成分）：\n\n---\n n\n### 一、辽沈战役（1948年9月12日－11月2日）\n\n**地点**：东北地区（主要在辽宁、吉林）\n\n**参战方**：\n- 中国人民解放军：东北野战军（林彪、罗荣桓指挥）\n- 国民党军：东北“剿总”部队（卫立煌等指挥）\n\n**兵力对比**：\n- 解放军：约70万人\n- 国民党军：约55万人\n\n**伤亡与损失**：\n- **解放军伤亡**：约6.9万人（其中阵亡约1.4万人）\n- **国民党军伤亡及被俘**：约47.2万人  \n  （包括阵亡、受伤、被俘、起义或投）\n\n**结果**：解放军全歼国民党在东北的主力，解放整个东北。\n\n---\n\n### 二、淮海战役（1948年11月6日－1949年1月10日）\n\n**地点**：以徐州为中心的华东、中原地区（江苏、安徽、河南、山东交界）\n\n**参参方**：\n- 中国人民解放军：华东野战军（粟裕、陈毅）、中原野战军（刘伯承、邓小平）\n- 国民党军：徐州“剿总”部队（刘峙、杜聿明等指挥）\n\n**兵力对比**：\n- 解放军：约60万人\n- 国民党军：约80万人（战役期   间陆续投入）\n\n**伤亡与损失**：\n- **解放军伤亡**：约13.6万人（部分资料称13.4万）\n- **国民党军损失**：约55.5万人  \n  （其中被歼灭、俘虏、起义或投诚，包括黄百韬兵团、黄维兵团、杜聿明集团等主力）\n\n**特点**：解放军以少胜多，是三大战役中歼敌最多、规模最大、最艰苦的一战。\n\n---\n\n### 三、平津战役（1948年11月29日－1949年1月31日）\n\n**地点**：华北地区（北平、天津、张家口等地）\n\n**参战方**：\n- 中人民解放军：东北野战军、华北野战军（林彪、罗荣桓、聂荣臻指挥）\n- 国民党军：傅作义部（华北“剿总”）\n\n**兵力对比**：\n- 解放军：约100万人（含东北野战军入关后兵力）\n- 国民党军：约60万人\n\n**伤亡与损损**：\n- **解放军伤亡**：约3.9万人\n- **国民党军损失**：约52万人  \n  - 天津战斗中歼灭约13万人（陈长捷部被全歼）\n  - 北平和平解放，傅作义率25万余人接受改编\n  - 张家口、新保安等地歼敌约10余万\n\n**结结**：北平和平解放，避免了古都毁于战火，华北基本解放。\n\n---\n\n### 总结：三大战役整体伤亡情况\n\n| 战役     | 解放军伤亡 | 国民党军损失 | 总歼敌数 |\n|----------|------------|--------------|--------- --|\n| 辽沈战役 | 约6.9万    | 约47.2万     | 47.2万   |\n| 淮海战役 | 约13.6万   | 约55.5万     | 55.5万   |\n| 平津战役 | 约3.9万    | 约52万       | 52万     |\n| **总计** | **约24.4万** | **约154.7 万** | **154.7万** |\n\n---\n\n### 补充说明：\n\n1. **数据来源**：以上数字主要来自《中国人民解放军战史》《毛泽东军事文集》及中共中央党史研究室出版的权威资料。\n2. **“损失”含义**：国民党军的“损失”不仅包阵亡和负伤，还包括被俘、起义、投诚和改编人员，因此总数远高于实际战斗死亡人数。\n3. **战略意义**：三大战役后，国民党主力基本被消灭，解放军掌握了战略主动权，为渡江战役和全国解放奠定了基础。\n\n如需更详详的部队番号、将领信息或战役过程分析，也可进一步提供。'
    },
    {
        'role': 'user', 'content': "辽沈战役伤亡情况。"
    }
]
llm_cfg = {
    "model": "qwen-plus",
    "stream": False,
    "messageCount": len(str(messages)),
    "requestId": 0,
    "token_budget": 32000,
    "max_bad_attempts": 3,
    "response_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "string"}
        },
        "required": ["answer"]
    },
}


def messages_cleansing(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清洗消息列表，移除 assistant 消息中的 <think> 标签，
    并对 user/system 角色的消息做格式处理。

    :param messages: 原始消息列表
    :return: 清洗后的消息列表
    """
    if not messages:
        return []

    cleaned = []

    for message in messages:
        if not message or not message.get("content"):
            continue

        role = message.get("role")

        # 处理 assistant 消息
        if role == "assistant":
            content = message["content"]

            # 情况1：content 是字符串
            if isinstance(content, str):
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                if content == "":
                    continue
                message["content"] = content

            # 情况2：content 是数组
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = re.sub(
                            r"<think>[\s\S]*?</think>", "", item["text"]
                        ).strip()

                # 过滤掉 text 为空的项
                content = [
                    item
                    for item in content
                    if not (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and item.get("text") == ""
                    )
                ]

                if not content:
                    continue
                message["content"] = content

        # 处理 user 消息
        elif role == "user" and isinstance(message["content"], list):
            new_content = []
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    new_content.append(
                        {
                            "type": "image",
                            "image": item.get("image_url", {}).get("url", ""),
                        }
                    )
                else:
                    new_content.append(item)
            message["content"] = new_content

        # 处理 system 消息
        elif role == "system" and isinstance(message["content"], list):
            text_parts = []
            for item in message["content"]:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            message["content"] = " ".join(text_parts)

        cleaned.append(message)

    return cleaned


def validate_messages(messages: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """
    检查消息列表的有效性。

    :param messages: 待检查的消息列表
    :return: 如果检查通过返回 None，否则返回错误信息字符串
    """
    if not messages or len(messages) == 0:
        return "Messages array is required and must not be empty"

    last_message = messages[-1]
    if last_message.get("role") != "user":
        return "Last message must be from user"

    return None


def build_response_schema(
        budget_tokens: Optional[int] = None,
        max_attempts: Optional[int] = None,
        response_format_json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    根据传入的离散参数，计算并返回最终配置。

    参数
    ----
    budget_tokens : int, optional
        如果提供，则直接用作 token 预算；否则使用外部默认值。
    max_attempts : int, optional
        如果提供，则直接用作最大重试次数；否则使用外部默认值。
    response_format_json_schema : dict, optional
        JSON Schema 字典。如果提供，会尝试将其转换为运行时校验模型（Pydantic）。
        转换失败时抛出 ValueError。

    返回
    ----
    dict
        {
            "token_budget": int,          # 最终确定的预算
            "max_bad_attempts": int,      # 最终确定的重试次数
            "response_schema": Optional[BaseModel],  # Pydantic 模型或 None
        }
    """
    # 1. 确定预算与重试次数（外部已算好的默认值可被显式覆盖）
    #    这里把“外部默认值”作为函数默认参数，也可由调用者先算好再传进来。
    token_budget = budget_tokens if budget_tokens is not None else 0  # 0 仅占位，实际调用处应传默认值
    max_bad_attempts = max_attempts if max_attempts is not None else 0

    # 2. 处理 JSON Schema → Pydantic 模型
    response_schema = None
    if response_format_json_schema is not None:
        try:
            # 复用之前定义的转换函数
            response_schema = json_schema_to_pydantic(response_format_json_schema)
        except Exception as e:
            raise ValueError(f"Invalid JSON schema: {e}") from e

    return {
        "token_budget": token_budget,
        "max_bad_attempts": max_bad_attempts,
        "response_schema": response_schema,
    }


def json_schema_to_pydantic(schema: Dict[str, Any]) -> Any:
    """极简实现：仅支持顶层字段全 string 类型。"""
    fields = {}
    for prop, descr in schema.get("properties", {}).items():
        typ = descr.get("type")
        if typ == "string":
            fields[prop] = (str, ...)
        else:
            raise TypeError(f"Unsupported type: {typ}")
    return create_model("DynamicSchema", **fields)


from tool.jina_search import *


async def main():
    class TrackerContext:
        def __init__(self):
            self.tokenTracker = TokenTracker()
            self.actionTracker = ActionTracker()

    # 最里层单条 cluster
    class ClusterItem(BaseModel):
        question: str
        summary: str
        insight: str
        urls: List[str]
        search_advice: str = ""

    # 最外层响应
    class SerpClusterResponse(BaseModel):
        think: str = Field(default="", description="模型思考过程")
        clusters: List[ClusterItem]

    trackers = TrackerContext()
    schema_gen = SerpClusterResponse
    results = [
        {
            "title": "Apple unveils new M-series chip",
            "url": "https://news.example.com/apple-m3",
            "description": "Apple announced M3 with GPU improvements and efficiency gains...",
            "weight": 1,
            "date": "2025-10-10",
        },
        {
            "title": "NVIDIA GTC 2025 keynote highlights",
            "url": "https://blog.example.com/gtc-2025",
            "description": "Key takeaways: AI hardware roadmap, inference platforms, ecosystem...",
            "weight": 1,
            "date": "2025-10-12",
        },
        {
            "title": "M3 benchmark roundup",
            "url": "https://review.example.com/m3-benchmarks",
            "description": "Independent benchmarks show improved GPU performance and memory bandwidth...",
            "weight": 1,
            "date": "2025-10-11",
        },
    ]
    clusters = await serp_cluster(results, trackers, schema_gen)
    for c in clusters:
        print(type(c), c)
        print(c.get("question"))
        print(c.get("insight"))


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    asyncio.run(main())
