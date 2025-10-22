import os
from typing import Optional

import instructor
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, conlist, constr, create_model
from sympy.physics.units import temperature

from utils.schemas import get_language_prompt

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


# def build_agent_action_model(allow_search=False, allow_visit=False):
#     annotations = {'think': constr(max_length=500), 'action': str}
#     attrs = {}
#     if allow_search:
#         class SearchRequest(BaseModel):
#             searchRequests: conlist(constr(min_length=1, max_length=30), min_length=1, max_length=5) = Field(
#                 ..., description="每个搜索请求字符串长度在1~30之间，数组最多4个请求。"
#             )
#
#         annotations['search'] = Optional[SearchRequest]
#         attrs['search'] = None
#     if allow_visit:
#         class VisitAction(BaseModel):
#             URLTargets: conlist(int, min_length=1, max_length=5) = Field(
#                 ..., description=f"一组URL索引，最多5个。"
#             )
#
#         annotations['visit'] = Optional[VisitAction]
#         attrs['visit'] = None
#
#     attrs['__annotations__'] = annotations
#     model = type("AgentActionDynamic", (BaseModel,), attrs)
#     return model


def build_agent_action_model(
        allow_search=True,
        allow_visit=True,
        allow_reflect=True,
        allow_read=True,
        allow_agent=True,
        allow_answer=True,
        allow_coding=True
):
    think_field = (
        constr(max_length=500),
        Field(..., description=f"Concisely explain your reasoning process in {get_language_prompt()}."),
    )
    action_fields = {}
    # annotation = {
    #     'think': (
    #         constr(max_length=500),
    #         Field(..., description=f"Concisely explain your reasoning process in {get_language_prompt()}."),
    #     ),
    #     'action': (
    #         dict,
    #         Field(
    #             ...,
    #             description="Show exactly all action from the available actions, "
    #                         "fill in the corresponding action schema required. "
    #                         "Keep the reasons in mind: "
    #                         "(1) What specific information is still needed? "
    #                         "(2) Why is this action most likely to provide that information? "
    #                         "(3) What alternatives did you consider and why were they rejected? "
    #                         "(4) How will this action advance toward the complete answer?"
    #         ),
    #     ),
    # }
    # 2. 启用的动作名
    enabled_actions: list[str] = []
    action_schemas = {}
    if allow_search:
        class SearchActionPayload(BaseModel):
            searchRequests: conlist(constr(min_length=1, max_length=30), max_length=5) = Field(
                min_length=1,
                max_length=30,
                description=(
                    "A Google search query.Based on the deep intention "
                    "behind the original question and the expected answer format."                "Required when action='search'. "
                    "Always prefer a single search query, "
                    "only add another search query if the original"
                    " question covers multiple aspects or elements and one search request is definitely not enough,"
                    " each request focus on one specific aspect of the original question. "
                    "Minimize mutual information between each query. "
                    f"Maximum 5 search queries."
                )
            )

        action_fields["search"] = (Optional[SearchActionPayload], None)

        # annotation["action"] = (Optional[SearchActionPayload], None)
        # # annotation["search"] = (Optional[SearchActionPayload], None)
        # enabled_actions.append("search")
        # action_schemas["search"] = SearchActionPayload

    if allow_coding:
        class CodingActionPayload(BaseModel):
            __doc__ = (
                "Required when action='coding'. "
                "Describe what issue to solve with coding, "
                "format like a github issue ticket. Specify the input value when it is short."
            )
            coding_issue: constr(max_length=500)

        action_fields["coding"] = (Optional[CodingActionPayload], None)
        # annotation["coding"] = (Optional[CodingActionPayload], None)
        # enabled_actions.append("coding")
        # action_schemas["coding"] = CodingActionPayload

    ActionModel = create_model(
        "ActionModel",
        __base__=BaseModel,
        **action_fields
        # **annotation,  # type: ignore
    )
    AgentActionDynamic = create_model(
        "AgentActionDynamic",
        __base__=BaseModel,
        think=think_field,
        action=(
            ActionModel,
            Field(
                ...,
                description="There are lots of actions below, I need u to choose nothing. search output None."
            ),
        ),
    )

    # 5. 把动作 schema 挂到类上，方便外部取用
    # AgentActionDynamic.__action_schemas__ = action_schemas
    return AgentActionDynamic


# 只允许 search
AgentActionOnlySearch = build_agent_action_model(allow_search=True, allow_coding=True)

obj, completion = wrapped.chat.completions.create_with_completion(
    model="qwen-plus",
    response_model=AgentActionOnlySearch,
    messages=messages,
    temperature=0
)

object_dict = obj.model_dump() if isinstance(obj, BaseModel) else obj
print(object_dict)
