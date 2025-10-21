import os
from typing import Optional

import instructor
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, conlist, constr
from sympy.physics.units import temperature

# 加载.env文件到环境变量
load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用ideaLAB的API Key将下行替换为：api_key="xxx",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

messages = [
    {"role": "system", "content": "你是一位历史研究专家。"},
    {"role": "user", "content": "你有哪些工具可用？"}
]

mode = instructor.Mode.TOOLS
wrapped = instructor.from_openai(client, mode=mode)



def build_agent_action_model(allow_search=False, allow_visit=False):
    annotations = {'think': constr(max_length=500), 'action': str}
    attrs = {}
    if allow_search:
        class SearchRequest(BaseModel):
            searchRequests: conlist(constr(min_length=1, max_length=30), min_length=1, max_length=5) = Field(
                ..., description="每个搜索请求字符串长度在1~30之间，数组最多4个请求。"
            )
        annotations['search'] = Optional[SearchRequest]
        attrs['search'] = None
    if allow_visit:
        class VisitAction(BaseModel):
            URLTargets: conlist(int, min_length=1, max_length=5) = Field(
                ..., description=f"一组URL索引，最多5个。"
            )

        annotations['visit'] = Optional[VisitAction]
        attrs['visit'] = None

    attrs['__annotations__'] = annotations
    model = type("AgentActionDynamic", (BaseModel,), attrs)
    return model
# 只允许 search
AgentActionOnlySearch = build_agent_action_model(allow_search=True, allow_visit=False)

obj, completion = wrapped.chat.completions.create_with_completion(
    model="qwen-plus",
    response_model=AgentActionOnlySearch,
    messages=messages,
    temperature=0.7
)

object_dict = obj.model_dump() if isinstance(obj, BaseModel) else obj
print(object_dict)


