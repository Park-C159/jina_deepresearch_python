from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict, Protocol

import hjson  # pip install hjson

from utils.get_log import get_logger
log = get_logger("safe_generator")

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

# ---- 用于替换的外部函数/对象（占位） ----
async def ai_generate_object(**kwargs) -> Dict[str, Any]:
    raise NotImplementedError  # 请替换为真实实现

def getModel(name: str) -> Any:
    return name  # 示例：真实实现应返回模型句柄

def getToolConfig(name: str) -> Any:
    class C:  # 简易容器
        maxTokens = None
        temperature = 0.0
    return C()


ToolName = str
Schema = Any  # 允许为 zod 等同物或 JSON Schema dict


# ---- 与 TS 保持一致的返回与参数类型 ----
class GenerateObjectResult(TypedDict):
    object: Any
    usage: LanguageModelUsage

class GenerateOptions(TypedDict, total=False):
    model: ToolName
    schema: Schema
    prompt: str
    system: str
    messages: List[Any]
    numRetries: int


class TokenTracker:
    """占位 TokenTracker"""
    def trackUsage(self, model: str, usage: LanguageModelUsage) -> None:
        pass


class ObjectGeneratorSafe:
    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        self.token_tracker = token_tracker or TokenTracker()

    # 公开方法，与 TS 的 generateObject 对齐
    async def generate_object(self, options: GenerateOptions) -> GenerateObjectResult:

        model: ToolName = options.get("model")  # type: ignore
        schema: Schema = options.get("schema")  # type: ignore
        prompt: Optional[str] = options.get("prompt")
        system: Optional[str] = options.get("system")
        messages: Optional[List[Any]] = options.get("messages")
        num_retries: int = options.get("numRetries", 0)

        if not model or schema is None:
            raise ValueError("Model and schema are required parameters")

        try:
            # 主调用
            result = await ai_generate_object(
                model=getModel(model),
                schema=schema,
                prompt=prompt,
                system=system,
                messages=messages,
                maxTokens=getToolConfig(model).maxTokens,
                temperature=getToolConfig(model).temperature,
            )
            usage = result.get("usage", {})
            self.token_tracker.trackUsage(model, usage)
            return {"object": result.get("object"), "usage": usage}

        except Exception as error:
            # 第一次兜底：手动解析错误输出
            try:
                error_result = await self._handle_generate_object_error(error)
                self.token_tracker.trackUsage(model, error_result["usage"])
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

                        fallback_result = await ai_generate_object(
                            model=getModel("fallback"),
                            schema=distilled_schema,
                            prompt=(
                                "Following the given JSON schema, extract the field from below: \n\n "
                                f"{failed_output}"
                            ),
                            temperature=getToolConfig("fallback").temperature,
                        )
                        usage = fallback_result.get("usage", {})
                        self.token_tracker.trackUsage("fallback", usage)
                        log.debug("Distilled schema parse success!")
                        return {
                            "object": fallback_result.get("object"),
                            "usage": usage,
                        }

                    except Exception as fallback_error:
                        # 最后一搏：对 fallback 的错误再做手动解析
                        try:
                            last_chance = await self._handle_generate_object_error(fallback_error)
                            self.token_tracker.trackUsage("fallback", last_chance["usage"])
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
