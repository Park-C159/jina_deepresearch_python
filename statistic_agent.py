"""
统计智能体 (Statistic Agent)
===========================
工作流：Plan → Confirm → Execute → Report

1. Plan:   接收用户问题，生成结构化分析计划（视角、指标、图表）
2. Confirm: 将 Plan 返回到前端，等待用户确认/调整
3. Execute: 按 Plan 执行搜索/数据提取/统计分析（回归、分类等）
4. Report:  生成包含可视化图表的 Markdown 统计报告，支持预览和下载
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent import TrackerContext, get_response
from plot_chart import AnalysisPlan as PlanSchema, AnalysisAngle, analysis_plan
from tool.analysis_sandbox import AnalysisCodeSandbox
from utils.safe_generator import ObjectGeneratorSafe
from utils.token_tracker import TokenTracker


class ExtractedDataSchema(BaseModel):
    """从文本中提取的结构化数据"""
    extracted_data: List[Dict[str, Any]] = Field(default_factory=list)
    time_series: List[Dict[str, Any]] = Field(default_factory=list)
    variables: Dict[str, List[Any]] = Field(default_factory=dict)
    notes: str = ""


# ---------- 三种分析模式 ----------
# 不同模式通过在「计划生成内容」前注入引导语，影响 analysis_plan 生成的视角方向，
# 从而让「报告生成 / 归因分析 / 数据生成」三个功能产生实际差异化的分析结果。

MODE_GUIDANCE = {
    # 报告生成：保持默认（综合多视角），不额外约束
    "report": "",
    # 归因分析：聚焦"指标为什么波动"，按维度下钻找主因
    "attribution": (
        "【分析类型：归因分析】\n"
        "请围绕「目标指标为什么会发生变化 / 波动」这一核心问题来规划分析视角。"
        "要求：选定关键指标后，从不同维度（如时间、品类、渠道、地区、用户群体等）"
        "逐层下钻，找出导致该指标上升或下降的主要影响因素，并尽量量化各因素的贡献；"
        "每个视角都应明确指向「某因素如何影响目标指标」，而非泛泛的现状描述。\n\n"
    ),
    # 数据生成：聚焦结构化提取与描述性统计，输出以数据表为主
    "data": (
        "【分析类型：数据生成 / 结构化提取】\n"
        "请聚焦于从文本 / 数据中提取结构化数据并进行描述性统计计算。"
        "视角应覆盖：关键字段的提取与汇总、分组聚合统计（计数、求和、均值、占比）、"
        "排序与 TopN、时间序列整理等，结果以清晰的数据表格为主要呈现形式。\n\n"
    ),
}

# 各模式下报告标题
MODE_TITLES = {
    "report": "统计分析报告",
    "attribution": "归因分析报告",
    "data": "数据提取与统计报告",
}

# 单次分析步骤喂给 LLM 的（联网检索）原文最大字符数。
# 注意：用户上传的数据不再走"让 LLM 回吐结构化数据"的路径，
# 而是落地为真实 CSV 由分析/绘图代码直接读取，因此无需为上传数据设置巨大的字符预算。
MAX_ANALYSIS_CHARS_SEARCH = 8000


@dataclass
class PlanStep:
    """计划中的单个步骤"""
    step_id: str
    step_type: str  # "search" | "extract" | "analyze" | "visualize"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | completed | failed
    result: Any = None
    error: Optional[str] = None


@dataclass
class StatisticPlan:
    """完整的分析计划"""
    plan_id: str
    question: str
    angles: List[AnalysisAngle]
    steps: List[PlanStep]
    estimated_charts: int
    estimated_tokens: int
    created_at: str
    # 用户上传的自定义数据（已解析为文本）。
    # 非空时，执行阶段将**跳过联网检索**，直接基于该数据进行分析与可视化。
    data_text: Optional[str] = None
    # 分析模式：
    #   "report"      —— 报告生成：综合多视角分析，产出图文并茂的完整报告（默认）
    #   "attribution" —— 归因分析：聚焦指标为何波动，按维度下钻找主因
    #   "data"        —— 数据生成：结构化提取 + 描述性统计，以数据表为主
    mode: str = "report"


@dataclass
class ExecutionProgress:
    """执行进度"""
    current_step: int
    total_steps: int
    step_type: str
    step_description: str
    status: str  # "running" | "completed" | "failed"
    logs: List[str] = field(default_factory=list)


class AnalystAgent:
    """分析团队成员（sub-agent）。

    每个 AnalystAgent 负责**一个分析视角**的完整链路：
        analyze（从文本提取结构化数据）→ visualize（生成统计图表）。

    多个 AnalystAgent 由 ``StatisticAgent`` 作为团队（agent team）并行调度，
    彼此独立、互不阻塞。其内部对 LLM / 绘图的调用复用 owner 的能力，
    LLM 请求经由 ``asyncio.to_thread`` 在线程池中真正并发执行。
    """

    def __init__(self, agent_id: int, steps: List[PlanStep], owner: "StatisticAgent"):
        self.agent_id = agent_id
        self.steps = steps  # 该视角对应的步骤（一个 analyze + 可选的 visualize）
        self.owner = owner

    async def run(self, raw_text: str, on_step_done) -> Dict[str, Any]:
        """串内顺序执行本视角的各步骤；不同 agent 之间则并行运行。

        :param raw_text: 共享的检索原文
        :param on_step_done: 每完成一步调用的回调 ``(agent_id, step) -> None``
        :return: 该 agent 的步骤结果与图表
        """
        local_results: List[Dict[str, Any]] = []
        local_charts: List[Dict[str, Any]] = []

        for step in self.steps:
            try:
                if step.step_type == "analyze":
                    result = await self.owner._execute_analysis(step, raw_text)
                    step.status = "completed"
                    step.result = result

                elif step.step_type == "visualize":
                    result = await self.owner._execute_visualization(step, raw_text)
                    step.status = "completed"
                    step.result = result
                    if result.get("file_name"):
                        local_charts.append({
                            "title": step.parameters.get("plot_title", ""),
                            "path": result["file_name"],
                            "insight": step.parameters.get("insight", ""),
                        })

                local_results.append({
                    "step_id": step.step_id,
                    "status": step.status,
                    "result": step.result,
                })
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                local_results.append({
                    "step_id": step.step_id,
                    "status": "failed",
                    "error": str(e),
                })

            # 通知调度者：该步骤已结束（用于实时进度反馈）
            try:
                on_step_done(self.agent_id, step)
            except Exception:
                pass

        return {
            "agent_id": self.agent_id,
            "results": local_results,
            "charts": local_charts,
        }


class StatisticAgent:
    """
    统计智能体：Plan → Execute → Report
    """

    def __init__(self, token_budget: int = 1000000, team_size: int = 4):
        self.token_budget = token_budget
        # 团队规模：同时并行工作的 AnalystAgent 数量上限（控制对 LLM 服务的并发压力）
        self.team_size = max(1, int(team_size))
        self.context = TrackerContext()
        self.context.tokenTracker = TokenTracker(token_budget)
        self.generator = ObjectGeneratorSafe(self.context.tokenTracker)
        self.sandbox = AnalysisCodeSandbox(trackers=self.context, maxAttempts=3)
        self._progress_callbacks: List = []
        self._current_plan: Optional[StatisticPlan] = None
        # 标记本次执行所用原文是否来自用户上传数据：
        # 为 True 时分析步骤会以更大的字符预算读取原文，确保上传内容被充分使用。
        self._data_from_upload: bool = False

    def on_progress(self, callback):
        """注册进度回调函数"""
        self._progress_callbacks.append(callback)

    def _emit_progress(self, progress: ExecutionProgress):
        """触发进度更新"""
        for cb in self._progress_callbacks:
            try:
                cb(progress)
            except Exception:
                pass

    async def generate_plan(self, question: str, data_text: Optional[str] = None,
                            mode: str = "report") -> StatisticPlan:
        """
        阶段 1: 根据用户问题生成分析计划 Plan
        仅基于问题本身生成分析视角，**不触发搜索**；
        搜索 / 数据提取 / 分析等后续操作延迟到 execute_plan（用户确认或编辑计划后）再执行。
        返回结构化计划，供前端展示和确认。

        :param question: 用户输入的分析主题 / 问题
        :param data_text: 可选。用户上传的自定义数据（已解析为文本）。
            若提供，则分析计划直接**基于真实数据内容**生成（视角会贴合数据中的字段与数值），
            且执行阶段会跳过联网检索，转而对该数据进行分析。
        :param mode: 分析模式 —— "report"（报告生成）/ "attribution"（归因分析）/ "data"（数据生成）。
            不同模式会通过引导语影响所生成视角的方向。
        """
        mode = mode if mode in MODE_GUIDANCE else "report"
        guidance = MODE_GUIDANCE.get(mode, "")

        # 计划生成的依据：
        # - 有上传数据时，把数据内容作为主要规划素材，并以用户问题作为分析侧重点引导；
        # - 无上传数据时，沿用原行为，仅基于问题本身规划（执行阶段再联网检索）。
        if data_text and data_text.strip():
            plan_content = (
                f"{guidance}"
                f"分析目标 / 用户关注点：{question}\n\n"
                f"以下是需要分析的真实数据：\n{data_text}"
            )
        else:
            plan_content = f"{guidance}{question}"

        angles = await analysis_plan(plan_content, self.context)

        # 构建执行步骤
        steps = []
        for idx, angle in enumerate(angles):
            step_id = f"step_{idx + 1}"
            steps.append(PlanStep(
                step_id=step_id,
                step_type="analyze",
                description=angle.get("insight", ""),
                parameters={
                    "think": angle.get("think", ""),
                    "plan": angle.get("plan", ""),
                    "need_plot": angle.get("need_plot", False),
                    "plot_title": angle.get("plot_title", ""),
                },
            ))
            if angle.get("need_plot"):
                steps.append(PlanStep(
                    step_id=f"{step_id}_viz",
                    step_type="visualize",
                    description=f"可视化: {angle.get('plot_title', 'chart')}",
                    parameters={
                        "plot_title": angle.get("plot_title", ""),
                        "insight": angle.get("insight", ""),
                        "plan": angle.get("plan", ""),
                    },
                ))

        plan = StatisticPlan(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            question=question,
            angles=angles if isinstance(angles, list) else [],
            steps=steps,
            estimated_charts=sum(1 for a in angles if isinstance(a, dict) and a.get("need_plot")),
            estimated_tokens=self.token_budget // 2,
            created_at=datetime.now().isoformat(),
            data_text=data_text if (data_text and data_text.strip()) else None,
            mode=mode,
        )
        self._current_plan = plan
        return plan

    def add_custom_angles(self, plan: StatisticPlan, custom_text: str) -> List[str]:
        """把用户补充的分析视角追加进计划。

        用户在前端输入的内容按行拆分，每一行（非空）作为一个新的分析视角，
        会同时生成对应的 analyze 步骤（如标注需要图表则附带 visualize 步骤）。

        :param plan: 当前计划
        :param custom_text: 用户输入的补充内容（支持多行，每行一条视角）
        :return: 新增的 step_id 列表（便于调用方将其纳入待执行步骤）
        """
        new_step_ids: List[str] = []
        if not custom_text or not custom_text.strip():
            return new_step_ids

        lines = [ln.strip() for ln in custom_text.splitlines() if ln.strip()]
        base_idx = len(plan.angles)
        for offset, line in enumerate(lines):
            angle = {
                "think": "用户补充的分析视角",
                "insight": line,
                "plan": line,
                "need_plot": True,
                "plot_title": line[:30],
            }
            plan.angles.append(angle)

            step_id = f"step_custom_{base_idx + offset + 1}"
            plan.steps.append(PlanStep(
                step_id=step_id,
                step_type="analyze",
                description=angle["insight"],
                parameters={
                    "think": angle["think"],
                    "plan": angle["plan"],
                    "need_plot": angle["need_plot"],
                    "plot_title": angle["plot_title"],
                },
            ))
            new_step_ids.append(step_id)

            viz_step_id = f"{step_id}_viz"
            plan.steps.append(PlanStep(
                step_id=viz_step_id,
                step_type="visualize",
                description=f"可视化: {angle['plot_title']}",
                parameters={
                    "plot_title": angle["plot_title"],
                    "insight": angle["insight"],
                    "plan": angle["plan"],
                },
            ))
            new_step_ids.append(viz_step_id)
            plan.estimated_charts += 1

        return new_step_ids

    async def execute_plan(self, plan: StatisticPlan, confirmed_steps: Optional[List[str]] = None,
                           use_search: bool = True) -> Dict[str, Any]:
        """
        阶段 3: 执行已确认的分析计划
        :param plan: 用户确认后的计划
        :param confirmed_steps: 用户确认的 step_id 列表，None 表示全部执行
        :param use_search: 无上传数据时是否联网检索。
            为 False 时**不进行检索**，直接基于问题描述/已知信息分析（适合不依赖外部资料的场景）。
        """
        steps_to_run = plan.steps
        if confirmed_steps:
            steps_to_run = [s for s in plan.steps if s.step_id in confirmed_steps]

        total = len(steps_to_run)
        results = {
            "plan_id": plan.plan_id,
            "question": plan.question,
            "steps_results": [],
            "charts": [],
            "report_md": "",
        }

        # ---- 确定分析原文 raw_text ----
        # 优先级：用户上传数据 > 联网检索 > 仅问题描述。
        # 只要存在上传数据，就**始终**以其作为分析原文并跳过联网检索（不受 use_search 影响）。
        if plan.data_text and plan.data_text.strip():
            # 用户上传了自定义数据：直接以其作为分析原文，跳过联网检索。
            self._data_from_upload = True
            self._emit_progress(ExecutionProgress(
                current_step=0, total_steps=total,
                step_type="data", step_description="优先使用用户上传的数据进行分析（跳过联网检索）...",
                status="running",
            ))
            raw_text = plan.data_text
        else:
            self._data_from_upload = False
            # 无上传数据：根据 use_search 决定是否联网检索。
            if use_search:
                self._emit_progress(ExecutionProgress(
                    current_step=0, total_steps=total,
                    step_type="search", step_description="正在检索相关数据...",
                    status="running",
                ))

                search_result = await get_response(
                    question=plan.question,
                    search_languge_code="zh",
                    search_provider=os.getenv("SEARCH_PROVIDER", "jina"),
                    language_code="zh",
                    with_images=False,
                    token_budget=self.token_budget,
                    max_bad_attempts=2,
                    existing_context=None,
                    messages=[],
                    num_returned_urls=10,
                    no_direct_answer=True,
                    max_ref=20,
                    min_rel_score=0.6,
                )

                answer_data = search_result.get("result", {})
                raw_text = answer_data.get("answer", "") or answer_data.get("mdAnswer", "")
            else:
                # 不检索：基于问题描述本身进行分析（不依赖外部资料）。
                self._emit_progress(ExecutionProgress(
                    current_step=0, total_steps=total,
                    step_type="data",
                    step_description="未启用联网检索，基于问题描述进行分析...",
                    status="running",
                ))
                raw_text = plan.question

        # ---- Agent Team 并行执行 ----
        # 将步骤按分析视角分组，每组交给一个 AnalystAgent，团队成员之间并行工作。
        groups = self._group_steps_by_angle(steps_to_run)
        agents = [AnalystAgent(idx + 1, grp, self) for idx, grp in enumerate(groups)]

        self._emit_progress(ExecutionProgress(
            current_step=0, total_steps=total,
            step_type="dispatch",
            step_description=f"组建分析团队：{len(agents)} 个分析师并行处理 {total} 个步骤（并发上限 {self.team_size}）",
            status="running",
        ))

        # 进度计数（在主事件循环线程内同步更新，无需加锁即安全）
        done_counter = {"n": 0}

        def on_step_done(agent_id: int, step: PlanStep):
            done_counter["n"] += 1
            self._emit_progress(ExecutionProgress(
                current_step=done_counter["n"], total_steps=total,
                step_type=step.step_type,
                step_description=f"[分析师 {agent_id}] {step.description}",
                status=step.status,
            ))

        # Semaphore 限制同时运行的 agent 数量，避免对 LLM 服务造成过大并发压力
        sem = asyncio.Semaphore(self.team_size)

        async def _run_agent(agent: AnalystAgent) -> Dict[str, Any]:
            async with sem:
                self._emit_progress(ExecutionProgress(
                    current_step=done_counter["n"], total_steps=total,
                    step_type="agent",
                    step_description=f"分析师 {agent.agent_id} 开始工作（{len(agent.steps)} 步）",
                    status="running",
                ))
                return await agent.run(raw_text, on_step_done)

        agent_outputs = await asyncio.gather(
            *[_run_agent(a) for a in agents], return_exceptions=True
        )

        # 按 agent 顺序聚合结果，保证报告中图表/结论顺序稳定
        for out in agent_outputs:
            if isinstance(out, Exception):
                results["steps_results"].append({
                    "step_id": "agent_error",
                    "status": "failed",
                    "error": str(out),
                })
                continue
            results["steps_results"].extend(out.get("results", []))
            results["charts"].extend(out.get("charts", []))

        # 生成最终报告
        self._emit_progress(ExecutionProgress(
            current_step=total, total_steps=total,
            step_type="report", step_description="正在生成统计报告...",
            status="running",
        ))

        report_md = await self._generate_report(raw_text, results["charts"], plan)
        results["report_md"] = report_md

        # 保存报告
        report_path = f"reports/report_{plan.plan_id}.md"
        os.makedirs("reports", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        results["report_path"] = report_path

        self._emit_progress(ExecutionProgress(
            current_step=total, total_steps=total,
            step_type="report", step_description="报告生成完成",
            status="completed",
        ))

        return results

    def _group_steps_by_angle(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """把待执行步骤按分析视角分组，作为每个 AnalystAgent 的任务单元。

        生成计划时步骤顺序为 ``[analyze, (visualize), analyze, (visualize), ...]``，
        其中 visualize 的 step_id 形如 ``"{analyze_step_id}_viz"``。
        这里以 analyze 步骤为组首，把紧随其后的 visualize 步骤归入同组；
        若某 visualize 找不到所属 analyze（例如用户只勾选了可视化），则单独成组。

        :return: 分组后的步骤列表，每个子列表交给一个 AnalystAgent
        """
        groups: List[List[PlanStep]] = []
        for step in steps:
            if step.step_type == "analyze" or not groups:
                groups.append([step])
            elif step.step_type == "visualize":
                # 归入最近一个组（其对应的 analyze）
                groups[-1].append(step)
            else:
                groups.append([step])
        return groups

    async def _execute_analysis(self, step: PlanStep, raw_text: str) -> Dict[str, Any]:
        """执行数据分析步骤"""
        params = step.parameters

        # 使用上传数据时：完整数据已落地为 CSV，且会由可视化步骤的代码直接读取。
        # 此处**不再让 LLM 回吐全量结构化数据**——否则大数据集会把数据塞进 tool_call 参数，
        # 极易触发 token 截断、JSON 非法、"function.arguments must be JSON" 等错误。
        # 因此上传数据场景下，分析步骤只做轻量记录，真正的数值计算交给绘图/统计代码。
        if self._data_from_upload:
            return {
                "extracted_data": {
                    "notes": "数据来自用户上传文件，完整数据已落地为 CSV，由分析/绘图代码直接读取，未在此步骤重复提取。",
                },
                "insight": params.get("insight", ""),
            }

        # 让 LLM 从文本中提取结构化数据（联网检索原文场景）。
        char_budget = MAX_ANALYSIS_CHARS_SEARCH
        prompt = f"""
你是一个数据提取专家。请从以下文本中提取与"{params.get('insight', '')}"相关的结构化数据。

分析计划：
{params.get('plan', '')}

原始文本：
{raw_text[:char_budget]}

请提取：
1. 所有相关数值数据
2. 分类标签和对应数值
3. 时间序列数据（如果有）
4. 任何适合回归或分类分析的变量对

返回 JSON 格式：
{{
  "extracted_data": [{{"label": "...", "value": ..., "category": "..."}}],
  "time_series": [{{"date": "...", "value": ...}}],
  "variables": {{"x": [...], "y": [...]}},
  "notes": "数据提取说明"
}}
"""

        res = await self.generator.generate_object({
            "model": "analysisPlan",
            "schema": ExtractedDataSchema,
            "prompt": prompt,
            "system": "你是一个数据提取和结构化专家，擅长从非结构化文本中提取数值数据。",
        })

        extracted = res.get("object", "{}")
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except Exception:
                extracted = {"raw": extracted}

        return {
            "extracted_data": extracted,
            "insight": params.get("insight", ""),
        }

    async def _execute_visualization(self, step: PlanStep, raw_text: str) -> Dict[str, Any]:
        """执行可视化步骤，生成统计图表（支持回归、分类等）"""
        params = step.parameters
        need = json.dumps({
            "plot_title": params.get("plot_title", ""),
            "insight": params.get("insight", ""),
            "plan": params.get("plan", ""),
        }, ensure_ascii=False, indent=2)

        safe_title = params.get("plot_title", "chart").replace(" ", "_").replace("/", "_")
        file_name = f"images/{safe_title}.png"

        result = await self.sandbox.solve(
            need=need,
            text=raw_text,
            file_name=file_name,
        )

        return {
            "success": result.get("solution") is not None,
            "file_name": file_name,
            "code": result.get("solution", {}).get("code", "") if result.get("solution") else "",
        }

    async def _generate_report(self, raw_text: str, charts: List[Dict], plan: StatisticPlan) -> str:
        """生成最终的 Markdown 统计报告"""
        from plot_chart import generate_final_report

        # 构建图表列表
        plot_list = charts

        # 调用报告生成
        report_md = await generate_final_report(raw_text, plot_list, self.context)

        # 添加统计报告头部信息
        report_title = MODE_TITLES.get(getattr(plan, "mode", "report"), "统计分析报告")
        header = f"""# {report_title}

> **分析主题**: {plan.question}
> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **分析视角数**: {len(plan.angles)}
> **生成图表数**: {len(charts)}

---

"""

        # 在报告开头插入统计摘要
        summary = self._generate_summary(plan, charts)

        return header + summary + report_md

    def _generate_summary(self, plan: StatisticPlan, charts: List[Dict]) -> str:
        """生成统计摘要"""
        summary = "## 分析摘要\n\n"
        summary += "| 分析视角 | 核心洞察 | 可视化 |\n"
        summary += "|---------|---------|--------|\n"

        for angle in plan.angles:
            if isinstance(angle, dict):
                summary += f"| {angle.get('insight', '')[:30]}... | {angle.get('plan', '')[:40]}... | {'是' if angle.get('need_plot') else '否'} |\n"

        summary += "\n### 生成图表\n\n"
        for chart in charts:
            summary += f"- **{chart.get('title', '')}**: {chart.get('insight', '')}\n"
            summary += f"  ![{chart.get('title', '')}]({chart.get('path', '')})\n\n"

        summary += "---\n\n"
        return summary


# -------------- 便捷入口 --------------

_agent_instance: Optional[StatisticAgent] = None


def get_statistic_agent(token_budget: int = 1000000) -> StatisticAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = StatisticAgent(token_budget)
    return _agent_instance


async def generate_statistic_plan(question: str, data_text: Optional[str] = None,
                                  mode: str = "report") -> StatisticPlan:
    """便捷函数：生成分析计划"""
    agent = get_statistic_agent()
    return await agent.generate_plan(question, data_text=data_text, mode=mode)


async def execute_statistic_plan(plan: StatisticPlan, confirmed_steps: Optional[List[str]] = None,
                                 use_search: bool = True) -> Dict[str, Any]:
    """便捷函数：执行分析计划"""
    agent = get_statistic_agent()
    return await agent.execute_plan(plan, confirmed_steps, use_search=use_search)
