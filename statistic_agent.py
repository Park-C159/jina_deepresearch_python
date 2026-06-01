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

from agent import TrackerContext, get_response
from plot_chart import AnalysisPlan as PlanSchema, AnalysisAngle, analysis_plan
from tool.analysis_sandbox import AnalysisCodeSandbox
from utils.safe_generator import ObjectGeneratorSafe
from utils.token_tracker import TokenTracker


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


@dataclass
class ExecutionProgress:
    """执行进度"""
    current_step: int
    total_steps: int
    step_type: str
    step_description: str
    status: str  # "running" | "completed" | "failed"
    logs: List[str] = field(default_factory=list)


class StatisticAgent:
    """
    统计智能体：Plan → Execute → Report
    """

    def __init__(self, token_budget: int = 1000000):
        self.token_budget = token_budget
        self.context = TrackerContext()
        self.context.tokenTracker = TokenTracker(token_budget)
        self.generator = ObjectGeneratorSafe(self.context.tokenTracker)
        self.sandbox = AnalysisCodeSandbox(trackers=self.context, maxAttempts=3)
        self._progress_callbacks: List = []
        self._current_plan: Optional[StatisticPlan] = None

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

    async def generate_plan(self, question: str) -> StatisticPlan:
        """
        阶段 1: 根据用户问题生成分析计划 Plan
        返回结构化计划，供前端展示和确认
        """
        # 先进行一轮浅层搜索，获取相关文本内容
        search_result = await get_response(
            question=question,
            search_languge_code="zh",
            search_provider=os.getenv("SEARCH_PROVIDER", "jina"),
            language_code="zh",
            with_images=False,
            token_budget=self.token_budget // 4,
            max_bad_attempts=1,
            existing_context=None,
            messages=[],
            num_returned_urls=5,
            no_direct_answer=True,
            max_ref=10,
            min_rel_score=0.7,
        )

        answer_data = search_result.get("result", {})
        raw_content = answer_data.get("answer", "") or answer_data.get("mdAnswer", "")

        # 生成分析视角计划
        angles = await analysis_plan(raw_content, self.context)

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
        )
        self._current_plan = plan
        return plan

    async def execute_plan(self, plan: StatisticPlan, confirmed_steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        阶段 3: 执行已确认的分析计划
        :param plan: 用户确认后的计划
        :param confirmed_steps: 用户确认的 step_id 列表，None 表示全部执行
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

        # 先重新搜索获取完整内容
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

        # 执行每一步
        for i, step in enumerate(steps_to_run):
            self._emit_progress(ExecutionProgress(
                current_step=i + 1, total_steps=total,
                step_type=step.step_type,
                step_description=step.description,
                status="running",
            ))

            try:
                if step.step_type == "analyze":
                    # 数据分析步骤：提取结构化数据
                    result = await self._execute_analysis(step, raw_text)
                    step.status = "completed"
                    step.result = result

                elif step.step_type == "visualize":
                    # 可视化步骤：生成统计图表
                    result = await self._execute_visualization(step, raw_text)
                    step.status = "completed"
                    step.result = result
                    if result.get("file_name"):
                        results["charts"].append({
                            "title": step.parameters.get("plot_title", ""),
                            "path": result["file_name"],
                            "insight": step.parameters.get("insight", ""),
                        })

                results["steps_results"].append({
                    "step_id": step.step_id,
                    "status": step.status,
                    "result": step.result,
                })

            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                results["steps_results"].append({
                    "step_id": step.step_id,
                    "status": "failed",
                    "error": str(e),
                })

            self._emit_progress(ExecutionProgress(
                current_step=i + 1, total_steps=total,
                step_type=step.step_type,
                step_description=step.description,
                status=step.status,
            ))

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

    async def _execute_analysis(self, step: PlanStep, raw_text: str) -> Dict[str, Any]:
        """执行数据分析步骤"""
        params = step.parameters
        # 构建分析需求
        need = json.dumps({
            "think": params.get("think", ""),
            "plan": params.get("plan", ""),
            "insight": params.get("insight", ""),
        }, ensure_ascii=False, indent=2)

        # 让 LLM 从文本中提取结构化数据
        prompt = f"""
你是一个数据提取专家。请从以下文本中提取与"{params.get('insight', '')}"相关的结构化数据。

分析计划：
{params.get('plan', '')}

原始文本：
{raw_text[:8000]}

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
            "schema": None,
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
        header = f"""# 统计分析报告

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


async def generate_statistic_plan(question: str) -> StatisticPlan:
    """便捷函数：生成分析计划"""
    agent = get_statistic_agent()
    return await agent.generate_plan(question)


async def execute_statistic_plan(plan: StatisticPlan, confirmed_steps: Optional[List[str]] = None) -> Dict[str, Any]:
    """便捷函数：执行分析计划"""
    agent = get_statistic_agent()
    return await agent.execute_plan(plan, confirmed_steps)
