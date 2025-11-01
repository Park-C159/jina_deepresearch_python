from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Protocol, Type, TypeVar, Coroutine
import hjson, openai, instructor
from instructor import Mode
from pydantic import BaseModel, Field
from config.config import get_tool_config, get_model, get_client
from utils.get_log import get_logger
from utils.token_tracker import TokenTracker

log = get_logger("safe_generator")

T = TypeVar("T", bound=BaseModel)


# ---- 这些依赖请替换为你工程中的真实实现 ----
# from ai import generate_object as ai_generate_object, LanguageModelUsage, NoObjectGeneratedError, Schema
# from yourpkg.token_tracker import TokenTracker
# from yourpkg.config import getModel, ToolName, getToolConfig
# from yourpkg.logging import logError, logDebug, logWarning

# 占位类型与协议（最小化约束）
class LanguageModelUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class SchemaDict(TypedDict, total=False):
    # JSON Schema 结构（子集）
    type: str
    properties: Dict[str, Any]
    items: Any
    anyOf: List[Any]
    allOf: List[Any]
    oneOf: List[Any]
    description: str


class NoObjectGeneratedErrorBase(Exception):
    """占位异常：需替换为 SDK 的 NoObjectGeneratedError"""

    def __init__(self, text: str, usage: Optional[LanguageModelUsage] = None):
        super().__init__("No object generated")
        self.text = text
        self.usage = usage or {}

    @classmethod
    def isInstance(cls, err: Exception) -> bool:
        return isinstance(err, cls)


# 协议：与 ai.generateObject 对齐（Python 侧假定返回 dict）
class GenerateObjectProtocol(Protocol):
    async def __call__(self, *, model: Any, schema: Any,
                       prompt: Optional[str] = None,
                       system: Optional[str] = None,
                       messages: Optional[List[Any]] = None,
                       maxTokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> Dict[str, Any]:
        ...


ToolName = str
Schema = Any  # 允许为 zod 等同物或 JSON Schema dict


# ---- 与 TS 保持一致的返回与参数类型 ----
class GenerateObjectResult(TypedDict):
    object: Any
    usage: LanguageModelUsage


class GenerateOptions(TypedDict, total=False):
    model: ToolName
    schema: Schema
    prompt: str | None
    system: str | None
    messages: List[Any] | None
    numRetries: int


def ai_generate_object(
        *,
        model,  # 对应 TS 的 model = get_model(model)
        schema=None,
        prompt=None,
        system=None,
        messages=None,
        maxTokens=4096 * 2,
        temperature=0.7,
):
    """
    把 TS 的 generateObject({ model, schema, prompt, system, messages, maxTokens, temperature })
    原封不动搬到 Python。
    """
    # 1. 拿到已绑定好 base_url / api_key 的「可 instructor 化」客户端
    #    这里假设你前面已经写好 get_openai_client()，返回 openai.OpenAI(...)
    # 映射表，按需扩展

    client, compatibility, model_name = get_model(model)


    _MODE_MAP = {
        "tools": Mode.TOOLS,  # function calling
        "json": Mode.JSON,  # JSON 模式
        "json_schema": Mode.JSON_SCHEMA,  # JSON Schema 模式
        "strict": Mode.MD_JSON,  # 最稳的 Markdown-JSON
    }

    mode = _MODE_MAP[compatibility] or instructor.Mode.TOOLS
    wrapped = instructor.from_openai(client, mode=mode)

    # 2. 组装 messages（优先级：messages > prompt+system）
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if prompt:
            # print(prompt)
            messages.append({"role": "user", "content": prompt})
        if not messages:
            raise ValueError("Either `messages` or (`prompt`/`system`) must be provided")

    if schema is None:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=maxTokens,
            temperature=temperature,
        )
        object_dict = completion.choices[0].message.content
    else:
        obj, completion = wrapped.chat.completions.create_with_completion(
            model=model_name,
            response_model=schema,
            messages=messages,
            max_tokens=maxTokens,
            temperature=temperature,
        )
        object_dict = obj.model_dump() if isinstance(obj, BaseModel) else obj
    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    try:
        finish_reason = getattr(completion.choices[0], "finish_reason", None)
    except AttributeError:
        finish_reason = None
    result: Dict[str, Any] = {
        "object": object_dict,
        "usage": {
            "promptTokens": input_tokens,
            "completionTokens": output_tokens,
            "totalTokens": total_tokens,
            "reasoningTokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens",
                                       None) if usage else None,
            "cachedInputTokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens",
                                         None) if usage else None,
        },
        "finishReason": finish_reason or "stop",
        "response": {
            "id": getattr(completion, "id", None),
            "modelId": getattr(completion, "model", model_name),
            "timestamp": (
                datetime.fromtimestamp(getattr(completion, "created", 0), tz=timezone.utc).isoformat()
                if getattr(completion, "created", None) is not None else None
            ),
            "headers": None,
            "body": None,
        },
        "reasoning": None,
        "warnings": None,
        "providerMetadata": None,
    }

    result["toJsonResponse"] = {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": {"object": result["object"]},
    }

    return result


class ObjectGeneratorSafe:
    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        self.token_tracker = token_tracker or TokenTracker()

    # 公开方法，与 TS 的 generateObject 对齐
    async def generate_object(self, options):
        model: ToolName = options.get("model")  # type: ignore
        schema: Schema = options.get("schema")  # type: ignore
        prompt: Optional[str] = options.get("prompt")
        system: Optional[str] = options.get("system")
        messages: Optional[List[Any]] = options.get("messages")
        num_retries: int = options.get("numRetries", 0)

        if not model or schema is None:
            raise ValueError("Model and schema are required parameters")

        try:
            result = ai_generate_object(
                model=model,
                schema=schema,
                prompt=prompt,
                system=system,
                messages=messages,
                maxTokens=get_tool_config(model).get('maxTokens'),
                temperature=get_tool_config(model).get('temperature'),
            )

            usage = result.get("usage", {})
            self.token_tracker.track_usage(model, usage)
            return {"object": result.get("object"), "usage": usage}

        except Exception as error:
            # print(error)
            # 第一次兜底：手动解析错误输出
            try:
                error_result = await self._handle_generate_object_error(error)
                self.token_tracker.track_usage(model, error_result["usage"])
                return error_result

            except Exception as parse_error:
                if num_retries > 0:
                    log.warning(
                        f"{model} failed on object generation -> manual parsing failed "
                        f"-> retry with {num_retries - 1} retries remaining"
                    )
                    new_opts: GenerateOptions = {
                        "model": model,
                        "schema": schema,
                        "prompt": prompt,
                        "system": system,
                        "messages": messages,
                        "numRetries": num_retries - 1,
                    }
                    return await self.generate_object(new_opts)
                else:
                    # 第二次兜底：使用 fallback 模型 + 蒸馏 schema
                    log.warning(
                        f"{model} failed on object generation -> manual parsing failed "
                        f"-> trying fallback with distilled schema"
                    )
                    try:
                        failed_output = ""
                        if isinstance(parse_error, NoObjectGeneratedErrorBase) or getattr(
                                type(parse_error), "isInstance", lambda _e: False
                        )(parse_error):
                            failed_output = getattr(parse_error, "text", "") or ""
                            log.debug(f"Failed output: {failed_output}")
                            cut = failed_output.rfind('"url":')
                            if cut != -1:
                                failed_output = failed_output[: min(cut, 8000)]
                            else:
                                failed_output = failed_output[:8000]

                        log.debug(f"Prompt: {failed_output}")

                        distilled_schema = self._create_distilled_schema(schema)

                        fallback_result = ai_generate_object(
                            model=get_model("fallback"),
                            schema=distilled_schema,
                            prompt=(
                                "Following the given JSON schema, extract the field from below: \n\n "
                                f"{failed_output}"
                            ),
                            temperature=get_tool_config("fallback").get("temperature"),
                        )
                        usage = fallback_result.get("usage", {})
                        self.token_tracker.track_usage("fallback", usage)
                        log.debug("Distilled schema parse success!")
                        return {
                            "object": fallback_result.get("object"),
                            "usage": usage,
                        }

                    except Exception as fallback_error:
                        # 最后一搏：对 fallback 的错误再做手动解析
                        try:
                            last_chance = await self._handle_generate_object_error(fallback_error)
                            self.token_tracker.track_usage("fallback", last_chance["usage"])
                            return last_chance
                        except Exception:
                            log.error("All recovery mechanisms failed")
                            # 抛出原始错误，便于调试
                            raise error

    # ---- 私有方法 ----
    def _create_distilled_schema(self, schema: Schema) -> Schema:
        """
        创建“蒸馏”版 schema：去除 description 字段。
        - 若传入的是 JSON Schema（dict），则深拷贝并递归删除 description。
        - 其他类型（例如第三方对象），无法可靠处理则原样返回。
        """
        if isinstance(schema, dict):
            cloned = json.loads(json.dumps(schema))  # 深拷贝
            self._remove_descriptions_in_schema_dict(cloned)
            return cloned
        # 这里可按需扩展对“zod 等价物”的识别与处理
        return schema

    def _remove_descriptions_in_schema_dict(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        # 删除本层 description
        if "description" in obj:
            del obj["description"]

        # 处理 properties
        props = obj.get("properties")
        if isinstance(props, dict):
            for k, v in list(props.items()):
                if isinstance(v, dict) and "description" in v:
                    del v["description"]
                self._remove_descriptions_in_schema_dict(v)

        # 处理 items
        items = obj.get("items")
        if items is not None:
            if isinstance(items, dict) and "description" in items:
                del items["description"]
            self._remove_descriptions_in_schema_dict(items)

        # 处理 anyOf / allOf / oneOf
        for key in ("anyOf", "allOf", "oneOf"):
            arr = obj.get(key)
            if isinstance(arr, list):
                for it in arr:
                    self._remove_descriptions_in_schema_dict(it)

    async def _handle_generate_object_error(self, error: Exception) -> GenerateObjectResult:
        """
        NoObjectGeneratedError -> 尝试 JSON / Hjson 宽松解析
        成功则返回 {"object": ..., "usage": ...}
        """
        # 兼容：既支持占位的 NoObjectGeneratedErrorBase，也支持 SDK 的 isInstance
        is_noobj = isinstance(error, NoObjectGeneratedErrorBase) or getattr(
            type(error), "isInstance", lambda _e: False
        )(error)
        if not is_noobj:
            raise error

        log.warning("Object not generated according to schema, fallback to manual parsing", {"error": str(error)})

        text = getattr(error, "text", "") or ""
        usage = getattr(error, "usage", {}) or {}

        # 先尝试 JSON
        try:
            partial = json.loads(text)
            log.debug("JSON parse success!")
            return {"object": partial, "usage": usage}
        except Exception:
            pass

        # 再尝试 Hjson
        try:
            partial = hjson.loads(text)
            log.debug("Hjson parse success!")
            return {"object": partial, "usage": usage}
        except Exception as hjson_error:
            log.error("Both JSON and Hjson parsing failed:", {"error": str(hjson_error)})
            # 抛回原始错误，交由上层处理
            raise error
