import asyncio
import logging
import datetime
import textwrap

from tool.code_sandbox import CodeSandbox


# 这里假设你已经有上面贴出来的 CodeSandbox / formatValue / analyzeStructure 定义
# 如果在同一个文件里，直接接在后面就行

# ---- 假生成器，替代 ObjectGeneratorSafe 用来测试 ----
class DummyGenerator:
    def __init__(self, token_tracker=None):
        self.token_tracker = token_tracker

    async def generate_object(self, payload):
        # 模拟模型返回结构：{"object": {"code": "..."}}
        code = """
result = 1 + 2
return result
"""
        return {
            "object": {
                "code": code
            }
        }


# ---- 假 trackers，避免 None 访问属性出错 ----
class DummyActionTracker:
    def trackThink(self, think):
        logging.info("trackThink called with:", think)


class DummyTrackers:
    def __init__(self):
        self.tokenTracker = None
        self.actionTracker = DummyActionTracker()


# ---- 实际测试 ----
async def main():
    logging.basicConfig(level=logging.INFO)

    trackers = DummyTrackers()
    sandbox = CodeSandbox(context={"foo": 123}, trackers=trackers, maxAttempts=3)

    # 用 DummyGenerator 替换掉真正的 ObjectGeneratorSafe
    sandbox.generator = DummyGenerator(trackers.tokenTracker)

    problem = {
        "description": "写一段 Python 代码，计算 1 + 2 并返回结果。",
        "language": "python"
    }

    result = await sandbox.solve(problem)

    print("=== solve 返回结果 ===")
    print(result)

    solution = result["solution"]
    attempts = result["attempts"]

    print("\n=== 最终代码 ===")
    print(solution["code"])

    print("\n=== 执行输出 ===")
    print(solution["output"])

    print("\n=== 失败尝试列表（如果有） ===")
    print(attempts)


if __name__ == "__main__":
    asyncio.run(main())
