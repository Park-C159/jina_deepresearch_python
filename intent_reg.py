import asyncio

from agent import TrackerContext
from tool.intent_reg import intent_query

if __name__ == "__main__":
    query = "三大战役伤亡情况？"
    think = (
        "用户在问中国内战中的三大战役的伤亡情况，需要给出双方的大致数据和说明。"
        "数字存在不同统计口径，需要说明是估算值。"
    )
    context = TrackerContext()
    intent = asyncio.run(intent_query(query, context))

    print("=== 识别出的意图结构 (Pydantic) ===")
    print(intent)
    # Pydantic 对象 -> dict -> 漂亮打印
    # print(json.dumps(intent.model_dump(), ensure_ascii=False, indent=2))
