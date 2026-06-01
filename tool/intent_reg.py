import logging

from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import ClarificationQuestionsSchema


def get_prompt(query):
    system_prompt = (
        "你是一个“意图澄清助手（Intent Clarifier）”。\n"
        "你的唯一任务是：根据用户当前输入，生成 3–5 个**简短、具体、有用**的澄清问题，帮助你在真正执行任务前把关键信息问清楚。\n\n"
        "要求：\n"
        "1. 所有问题用中文表述。\n"
        "2. 每个问题尽量围绕：主题范围、时间范围、潜在要求等关键点。\n"
        "3. 需要根据用户需要的内容挖掘主题背后深层的内容。\n"
        "4. 问题要尽量封闭式或半封闭式，避免太宽泛的‘还有什么要求？’之类空洞问题。\n"
        "5. 不要回答问题，只负责提问。\n"
        "6. 总是生成 3–5 个问题，每个问题之间尽可能相互正交。\n"
        "7. 目前只支持文字相关回复，问题不要关于输出形式。"
    )

    user_content_parts = [f"用户当前输入：{query}"]

    return {
        'system': system_prompt,
        'user': user_content_parts
    }


TOOL_NAME = 'intent_reg'


async def intent_query(query, trackers):
    try:
        generator = ObjectGeneratorSafe(trackers.tokenTracker)
        prompt = get_prompt(query)
        result = await generator.generate_object({
            'model': TOOL_NAME,
            'schema': ClarificationQuestionsSchema,
            'system': prompt.get('system'),
            'prompt': prompt.get('user'),
        })
        return result.get("object")['clarification_questions']

    except Exception as e:
        logging.error(e)
        return []
