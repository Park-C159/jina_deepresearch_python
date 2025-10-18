from typing import Any, Dict, List, TypedDict

from utils.get_log import get_logger
from utils.safe_generator import ObjectGeneratorSafe

TOOL_NAME = "serpCluster"
log = get_logger("serp_cluster")
# ---- 类型占位（可替换为你项目中的真实类型） ----
class SearchSnippet(TypedDict, total=False):
    title: str
    url: str
    description: str
    weight: int
    date: str

class PromptPair(TypedDict):
    system: str
    user: str

def get_prompt(results: List[SearchSnippet]) -> PromptPair:
    return {
        "system": (
            "You are a search engine result analyzer. You look at the SERP API "
            "response and group them into meaningful cluster.\n\n"
            "Each cluster should contain a summary of the content, key data and "
            "insights, the corresponding URLs and search advice. Respond in JSON format."
        ),
        "user": f"\n{__import__('json').dumps(results)}\n",
    }


async def serp_cluster(
    results: List[SearchSnippet],
    trackers: Any,        # TrackerContext
    schema_gen: Any,      # Schemas
) -> List[Dict[str, Any]]:
    """
    用来把一批搜索结果（SERP items）自动分成若干“主题簇（cluster）”，并为每个簇生成摘要、洞见、链接清单和后续检索建议。
    """
    try:
        generator = ObjectGeneratorSafe(trackers.tokenTracker)
        prompt = get_prompt(results)
        result = await generator.generate_object({
            "model": TOOL_NAME,
            "schema": schema_gen,
            "system": prompt["system"],
            "prompt": prompt["user"],
        })

        # 追踪思考
        obj = result.get("object", {})
        trackers.actionTracker.track_think(obj.get("think"))

        clusters = obj.get("clusters", [])
        log.info(TOOL_NAME, {"clusters": clusters})
        return clusters

    except Exception as error:
        log.error(TOOL_NAME, {"error": str(error)})
        raise
