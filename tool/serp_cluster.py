from typing import Any, Dict, List, TypedDict

from utils.get_log import get_logger
from utils.safe_generator import ObjectGeneratorSafe

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

TOOL_NAME = "serpCluster"

async def serp_cluster(
    results: List[SearchSnippet],
    trackers: Any,        # TrackerContext
    schema_gen: Any,      # Schemas
) -> List[Dict[str, Any]]:
    """
    Python 版 serpCluster：调用对象生成器根据 schema 产出聚类结果，
    并追踪 'think'，返回 clusters 列表。
    """
    try:
        generator = ObjectGeneratorSafe(trackers.tokenTracker)
        prompt = get_prompt(results)
        result = await generator.generate_object({
            "model": TOOL_NAME,
            "schema": schema_gen.getSerpClusterSchema(),
            "system": prompt["system"],
            "prompt": prompt["user"],
        })

        # 追踪思考
        obj = result.get("object", {})
        trackers.actionTracker.trackThink(obj.get("think"))

        clusters = obj.get("clusters", [])
        log.info(TOOL_NAME, {"clusters": clusters})
        return clusters

    except Exception as error:
        log.error(TOOL_NAME, {"error": str(error)})
        raise
