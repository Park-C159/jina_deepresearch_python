import logging
import os
import textwrap

import matplotlib.pyplot as plt

from utils.safe_generator import ObjectGeneratorSafe
from utils.schemas import CodeGeneratorSchema


def get_analysis_code_prompt(need, text, previous_attempts, file_name):
    """
    增强版：支持将之前失败的代码和错误信息一并给到模型，帮助它自我修正。
    previous_attempts: [{ "code": str, "error": str }, ...]
    """
    previous_attempts_arr = []
    for index, attempt in enumerate(previous_attempts):
        previous_attempt_str = f"""
<bad-attempt-${index + 1}>
{attempt.get('code')}
{'Error: ' + str(attempt.get('error')) if attempt.get('error') else attempt.get('error')}
</bad-attempt-${index + 1}>
    """
        previous_attempts_arr.append(previous_attempt_str)
    previous_attempts_context = '\n'.join(previous_attempts_arr)

    # 如果有历史失败，就在 system 里专门说明，让模型避免重复错误
    previous_block = ""
    if previous_attempts_context:
        previous_block = f"""
下面是之前生成但执行失败的代码及其错误，请务必分析并修正这些问题，不要重复犯同样的错误：
{previous_attempts_context}
"""

    return {
        'system': f"""你是一个Python统计分析代码生成助手，擅长根据文本和数据表结构，生成可直接运行的 Python + matplotlib 的统计可视化代码。

<rules>
- 生成直接绘图的纯 Python 代码。
- 一个图表规格，描述了要画的图类型、横轴维度、需要展示的度量等信息。
- 生成一段可直接执行的 Python 代码字符串，使用 `matplotlib` 根据下面本文中的信息和要求绘制指定图表，并保存为 PNG 图片。
- 你不需要也不能访问其余数据，只需要按文本中的内容抽取数据作为输入写出通用的绘图逻辑。
- 只能使用标准库 + `matplotlib` + `pandas`，不要使用 seaborn 等其他第三方可视化库。
- 默认假设运行环境中已经有：
```python
import matplotlib.pyplot as plt
```
</rules>

{'Previous attempts and their errors:' + str(previous_attempts_context) if len(previous_attempts) > 0 else ''}

图形要求：
- 设置合适的图像尺寸，例如 plt.figure(figsize=(8, 5))。
- 设置标题：使用 ChartSpec.title。
- 设置 x/y 轴标签（如果可以从上下文推断单位，可简单写入）。
- 添加图例，避免歧义。
- 使用 plt.tight_layout() 优化布局。
- 最后使用 plt.savefig("{file_name}", dpi=300, bbox_inches="tight") 保存图片，并调用 plt.close() 释放资源。
- 代码中不要调用 plt.show()，也不要打印多余信息。

{'之前的尝试及其错误：' + str(previous_attempts_context) if len(previous_attempts) > 0 else ''}
""",
        'user': f"""下面是我需要对数据进行分析的要求：
<need>
{need}
</need>

下面是原始包含有数据的文本内容：
<text>
{text}
</text>
"""
    }


chinese_patch = """
import matplotlib.pyplot as plt
import matplotlib
import platform, os, subprocess, urllib.request, tempfile, shutil

font_name = 'SimHei.ttf' if platform.system() == 'Windows' else 'SimHei.ttf'
font_url  = 'https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf'

temp_dir = tempfile.gettempdir()
font_path = os.path.join(temp_dir, font_name)
if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)

from matplotlib import font_manager
font_manager.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

matplotlib.rcParams['axes.unicode_minus'] = False
"""


class AnalysisCodeSandbox:
    """
    用于“从文本 + 图表需求生成 matplotlib 绘图代码并执行”的沙箱。

    特点：
    - 基于 ObjectGeneratorSafe 调用大模型生成代码（schema 使用 CodeGeneratorSchema，返回 {"code": "..."}）。
    - 支持多次重试（maxAttempts），每次将之前的代码和错误作为提示信息传给模型。
    - evaluateCode 中直接 exec 代码，代码中只需要调用 plt 画图并保存，不要求 return。
    - 默认把图片保存到 default_dir 指定的目录。
    """

    def __init__(self, trackers=None, maxAttempts: int = 3, default_dir: str = "plot"):
        """
        :param trackers: 上下文 tracker，内部会从中拿 tokenTracker / actionTracker（如果有的话）
        :param maxAttempts: 最多尝试生成 + 执行代码的次数
        :param default_dir: 默认图片保存目录
        """
        self.trackers = trackers
        token_tracker = trackers.tokenTracker if trackers else None

        # 兼容外部依赖命名：ObjectGeneratorSafe 必须由外部提供
        self.generator = ObjectGeneratorSafe(token_tracker)

        self.maxAttempts = maxAttempts
        self.default_dir = default_dir

        # 使用已有的 CodeGeneratorSchema，要求返回 {"code": "..."}
        self.schemaGen = CodeGeneratorSchema

    async def generateCode(self, need: str, text: str, file_name: str, previousAttempts=None):
        """
        调用大模型生成绘图代码，返回对象中应包含 code 字段。

        :param need: 图表需求说明（建议用 JSON 字符串，比如 ChartSpec 序列化后的结果）
        :param text: 原始包含数据的文本内容（比如 Markdown 报告）
        :param file_name: 期望在代码中使用的保存图片文件名，用于写进 prompt 中
        :param previousAttempts: 之前失败的尝试 [{'code': str, 'error': str}, ...]
        """

        if previousAttempts is None:
            previousAttempts = []

        # 这里假设你在别处实现了 get_analysis_code_prompt(need, text, previousAttempts, file_name)
        # 返回 {"system": "...", "user": "..."}
        prompt = get_analysis_code_prompt(need, text, previousAttempts, file_name)

        # 兼容方法命名：generate_object / generateObject
        gen_method = getattr(self.generator, "generate_object", None) or getattr(
            self.generator, "generateObject", None
        )
        if not gen_method:
            raise RuntimeError("Code generator method not found (generate_object / generateObject).")

        result = await gen_method(
            {
                "model": "analysisCoder",  # 模型路由名称，按你自己的配置修改
                "schema": self.schemaGen,
                "system": prompt.get("system"),
                "prompt": prompt.get("user"),
            }
        )

        # 记录思考轨迹（若可用）
        action_tracker = getattr(self.trackers, "actionTracker", None) if self.trackers else None
        track_think = (
            getattr(action_tracker, "track_think", None)
            or getattr(action_tracker, "trackThink", None)
            if action_tracker
            else None
        )
        if track_think and isinstance(result, dict):
            obj = result.get("object")
            think = (obj or {}).get("think") if isinstance(obj, dict) else None
            track_think(think)

        # 返回对象部分
        obj = result.get("object") if isinstance(result, dict) else None
        return obj

    def evaluateCode(self, code: str):
        """
        执行生成的绘图代码：
        - 不要求有 return。
        - 执行过程中不抛异常即认为成功。
        - 默认提供 plt 环境。
        """
        try:
            exec_env = {
                "plt": plt,
            }

            logging.debug("Running analysis code:\n%s", code)
            code = chinese_patch + code
            exec(code, exec_env, exec_env)

            return {
                "success": True,
                "error": None,
            }
        except Exception as error:
            return {
                "success": False,
                "error": str(error) if str(error) else "Unknown error occurred",
            }

    async def solve(self, need: str, text: str, file_name: str | None = None):
        """
        根据 need + text 生成绘图代码并执行，带重试机制。

        :param need: 图表需求说明（JSON 字符串或其它结构化描述）
        :param text: 原始数据文本
        :param file_name: 图片文件名。如果未提供，则自动生成到 default_dir 下。

        :return: {
            "solution": {
                "code": <最终成功的代码字符串>,
                "file_name": <保存图片的文件名>
            },
            "attempts": [
                {"code": ..., "error": ...},
                ...
            ]
        }
        """
        # 准备文件名和目录
        if not file_name:
            os.makedirs(self.default_dir, exist_ok=True)
            file_name = os.path.join(self.default_dir, "analysis_plot.png")
        else:
            dir_name = os.path.dirname(file_name)
            if not dir_name:
                os.makedirs(self.default_dir, exist_ok=True)
                file_name = os.path.join(self.default_dir, file_name)
            else:
                os.makedirs(dir_name, exist_ok=True)

        attempts = []

        for i in range(self.maxAttempts):
            # 1. 生成代码
            generation = await self.generateCode(need, text, file_name, attempts)

            code = (generation or {}).get("code") if isinstance(generation, dict) else None

            # 防御：若生成阶段没有返回 code
            if not code:
                error_msg = "No code was generated"
                logging.warning("Analysis coding error: %s", {"error": error_msg})
                attempts.append({"code": "", "error": error_msg})
                if i == self.maxAttempts - 1:
                    raise RuntimeError(
                        f"Failed to generate working analysis code after {self.maxAttempts} attempts"
                    )
                continue

            # 2. 执行并评估
            result = self.evaluateCode(code)

            if result.get("success"):
                logging.info(
                    "Analysis coding success: %s",
                    {"file_name": file_name},
                )
                return {
                    "solution": {
                        "code": code,
                        "file_name": file_name,
                    },
                    "attempts": attempts,
                }

            # 3. 记录失败并继续迭代
            logging.warning("Analysis coding error: %s", {"error": result.get("error")})
            attempts.append(
                {
                    "code": code,
                    "error": result.get("error"),
                }
            )

            if i == self.maxAttempts - 1:
                raise RuntimeError(
                    f"Failed to generate working analysis code after {self.maxAttempts} attempts"
                )

        # 理论上不会到这里
        raise RuntimeError("Unexpected end of analysis code execution")
