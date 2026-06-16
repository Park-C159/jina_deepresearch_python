import asyncio
import base64
import mimetypes
import os
import re
import time
import zipfile
from typing import Any, List, Optional

# ---------- 修复 gradio_client 1.4.0 的已知 bug ----------
# 当某组件 schema 中出现布尔型 additionalProperties（如 Dict[str, Any]）时，
# gradio_client.utils.get_type 会对 bool 执行 `"const" in schema` 而抛
# `TypeError: argument of type 'bool' is not iterable`，导致 /info 接口 500。
# 这里对其做最小化兜底补丁，遇到布尔型 schema 直接返回 "Any"。
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

import gradio as gr

from core.plugin_manager import get_manager
from statistic_agent import StatisticAgent, StatisticPlan


# ---------- 辅助函数 ----------

def _get_providers(category: str):
    try:
        mgr = get_manager()
        plugins = mgr.list(category)
        return list(plugins.keys()) if plugins else []
    except Exception:
        return []


# 各搜索提供商的默认地址（仅作前端占位提示用）
_SEARCH_DEFAULT_URLS = {
    "jina": "https://svip.jina.ai/",
    "milvus": "http://192.168.12.162:5445/milvus/rerank_query",
    "brave": "https://api.search.brave.com/res/v1/web/search",
}

# 中间区域的示例任务（点击即可填入输入框）
_EXAMPLE_TASKS = [
    "订单总实际支付金额变化趋势分析",
    "各一级品类总实际支付金额分布",
    "2024年中国新能源汽车销量趋势及竞争格局分析",
]


# Markdown 图片语法： ![alt](path "可选标题")
_MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^)]+?)\s*\)")


def _parse_img_path(raw: str) -> str:
    """从 Markdown 图片链接中提取出纯文件路径（去掉可选标题和引号）。"""
    raw = raw.strip()
    # 形如 path "title"：取第一个空白前的部分（路径本身不含空格时）
    if '"' in raw or "'" in raw:
        raw = raw.split('"')[0].split("'")[0]
    return raw.strip().strip('"').strip("'").strip()


def _embed_images_base64(md_content: str, base_dir: str = ".") -> str:
    """把 Markdown 中引用的本地图片替换为 base64 data URI。

    gr.Markdown 默认无法渲染相对/本地路径图片（浏览器找不到文件会显示裂图），
    转为内嵌 data URI 后即可在前端正常显示。网络图片与已内嵌的 data URI 保持原样。
    """
    if not md_content:
        return md_content

    def _repl(m):
        alt, path = m.group(1), m.group(2).strip()
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        raw_path = _parse_img_path(path)
        abs_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
        if not os.path.exists(abs_path):
            return m.group(0)
        mime, _ = mimetypes.guess_type(abs_path)
        mime = mime or "image/png"
        try:
            with open(abs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            return m.group(0)
        return f"![{alt}](data:{mime};base64,{b64})"

    return _MD_IMG_RE.sub(_repl, md_content)


def _collect_md_image_paths(md_content: str, base_dir: str = ".") -> List[str]:
    """收集 Markdown 中引用且存在于本地的图片绝对路径。"""
    paths: List[str] = []
    for m in _MD_IMG_RE.finditer(md_content or ""):
        path = m.group(2).strip()
        if path.startswith(("http://", "https://", "data:")):
            continue
        raw_path = _parse_img_path(path)
        abs_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
        abs_path = os.path.normpath(abs_path)
        if os.path.exists(abs_path) and abs_path not in paths:
            paths.append(abs_path)
    return paths


def _make_report_zip(report_path: str, base_dir: str = ".") -> Optional[str]:
    """把报告 .md 及其引用的本地图片一起打包为 zip，返回 zip 路径。

    图片以相对 base_dir 的路径（如 ``images/xxx.png``）写入压缩包，
    使解压后 Markdown 中的相对图片链接仍然有效。
    """
    if not report_path or not os.path.exists(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            md_content = f.read()
    except Exception:
        return None

    img_paths = _collect_md_image_paths(md_content, base_dir)
    zip_path = os.path.splitext(report_path)[0] + ".zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(report_path, arcname=os.path.basename(report_path))
            for p in img_paths:
                try:
                    rel = os.path.relpath(p, base_dir)
                except ValueError:
                    rel = os.path.basename(p)
                zf.write(p, arcname=rel)
    except Exception:
        return None
    return zip_path


def _apply_runtime_config(search_p, search_url, search_key, llm_p, llm_url, llm_key):
    """把前端选择的 provider 及自定义 url/api_key 应用到运行时，
    并清理 / 重建对应插件实例缓存，确保新配置即时生效。"""
    os.environ["SEARCH_PROVIDER"] = (search_p or "").strip()
    os.environ["LLM_PROVIDER"] = (llm_p or "").strip()

    def _set_or_clear(key, value):
        if value and value.strip():
            os.environ[key] = value.strip()
        else:
            os.environ.pop(key, None)

    # LLM 自定义 URL / API key（由 _get_llm_plugin 读取并传入插件）
    _set_or_clear("LLM_BASE_URL", llm_url)
    _set_or_clear("LLM_API_KEY", llm_key)

    # 重置 LLM 缓存，让新配置生效
    try:
        from config.config import _llm_plugin_cache
        _llm_plugin_cache.clear()
    except Exception:
        pass

    # 搜索引擎自定义 URL / API key：直接以显式配置重建插件实例
    # （jina / milvus / brave 等所有插件都优先读取 cfg 中的 base_url / api_key）
    try:
        from core.plugin_manager import reload_plugin
        if search_p:
            search_cfg = {}
            if search_url and search_url.strip():
                search_cfg["base_url"] = search_url.strip()
            if search_key and search_key.strip():
                search_cfg["api_key"] = search_key.strip()
            reload_plugin("search", search_p.strip(), config=search_cfg)
    except Exception:
        pass


# ---------- 自定义样式（模拟 Tomoro 三栏式现代界面） ----------

CUSTOM_CSS = """
/* 整体铺满，去掉 Gradio 默认窄容器 */
.gradio-container { max-width: 100% !important; padding: 0 !important; background: #f5f6f8; }
footer { display: none !important; }

#app-shell { gap: 0 !important; min-height: 100vh; }

/* ===== 左侧导航栏 ===== */
#sidebar {
    background: #f0f2f5;
    border-right: 1px solid #e6e8eb;
    padding: 16px 14px !important;
    min-width: 220px;
    max-width: 240px;
}
#brand {
    font-size: 20px; font-weight: 700; color: #1f2329;
    display: flex; align-items: center; gap: 8px; margin-bottom: 18px;
}
#brand .dot { color: #4f7cff; }
#sidebar .nav-block { margin-top: 8px; }
#sidebar .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 8px; color: #495057;
    font-size: 14px; cursor: default; user-select: none;
}
#sidebar .nav-item:hover { background: #e8ebf0; }
#sidebar .nav-title {
    font-size: 12px; color: #9aa0a6; margin: 18px 0 6px 4px; font-weight: 600;
}
#new-task-btn button {
    background: #ffffff !important; border: 1px solid #d7dbe0 !important;
    color: #1f2329 !important; font-weight: 600 !important; border-radius: 10px !important;
}

/* ===== 中间工作区 ===== */
#main-area { padding: 28px 36px !important; }
#hero { text-align: center; margin: 10px 0 22px; }
#hero h1 { font-size: 30px; font-weight: 700; color: #1f2329; margin: 6px 0; }
#hero p { color: #8a9099; font-size: 14px; }

.feature-cards { display: flex; gap: 14px; justify-content: center; margin: 18px 0 26px; flex-wrap: wrap; }
.feature-card {
    flex: 1; min-width: 150px; max-width: 220px; background: #fff;
    border: 1px solid #ebedf0; border-radius: 14px; padding: 16px;
    box-shadow: 0 2px 10px rgba(20,30,60,.04);
}
.feature-card .ic { font-size: 22px; }
.feature-card h4 { margin: 8px 0 4px; font-size: 15px; color: #1f2329; }
.feature-card span { font-size: 12px; color: #8a9099; }

/* 输入卡片 */
#compose-card {
    background: #fff; border: 1px solid #e6e8eb; border-radius: 16px;
    padding: 14px 16px !important; box-shadow: 0 4px 18px rgba(20,30,60,.06);
}
#compose-toolbar { align-items: center; }
#send-btn button {
    background: linear-gradient(135deg,#4f7cff,#6a5cff) !important; color: #fff !important;
    border: none !important; border-radius: 12px !important; font-weight: 600 !important;
}

/* ===== 右侧数据 / 报告面板 ===== */
#data-panel {
    background: #fff; border-left: 1px solid #e6e8eb;
    padding: 20px 22px !important; min-width: 320px;
}
#data-panel .panel-title { font-size: 15px; font-weight: 700; color: #1f2329; margin-bottom: 10px; }

.section-title { font-weight: 700; color: #1f2329; }
"""


# ---------- Gradio 界面 ----------

def create_ui():
    search_providers = _get_providers("search") or ["jina", "milvus"]
    llm_providers = _get_providers("llm") or ["openai", "gemini", "qwen"]

    with gr.Blocks(title="统计智能体 - Statistic Agent", theme=gr.themes.Soft(),
                   css=CUSTOM_CSS) as demo:

        # 状态变量
        current_plan = gr.State(None)
        report_file = gr.State(None)

        with gr.Row(elem_id="app-shell", equal_height=False):

            # ==================== 左侧导航栏 ====================
            with gr.Column(scale=1, elem_id="sidebar", min_width=220):
                gr.HTML('<div id="brand">📊 DeepStatic<span class="dot">·</span></div>')
                new_task_btn = gr.Button("+ 新任务", elem_id="new-task-btn")

            # ==================== 中间工作区 ====================
            with gr.Column(scale=3, elem_id="main-area"):
                gr.HTML(
                    '<div id="hero">'
                    '<p>Bring Data to <b style="color:#4f7cff;">DeepStatic</b></p>'
                    '<h1>统计智能体</h1>'
                    '<p>选择功能 → 生成分析计划 → 确认 → 并行执行 → 生成结果</p>'
                    '</div>'
                )

                # ---- 功能选择（三选一，均有实际支持）----
                mode_radio = gr.Radio(
                    choices=[
                        ("📈 报告生成 · 综合多视角图文分析报告", "report"),
                        ("🔎 归因分析 · 分析指标波动 / 变化的原因", "attribution"),
                        ("🗃️ 数据生成 · 结构化提取与描述性统计", "data"),
                    ],
                    value="report",
                    label="选择功能",
                )

                # ---- 输入卡片 ----
                with gr.Group(elem_id="compose-card"):
                    question_input = gr.Textbox(
                        label="分析主题 / 问题",
                        placeholder="例如：2024年中国新能源汽车销量趋势及竞争格局分析",
                        lines=3,
                    )
                    data_files_input = gr.File(
                        label="上传自定义数据（可选，CSV / Excel / JSON / TXT / Markdown，可多选）"
                              "；上传后将基于这些数据进行分析，并跳过联网检索",
                        file_count="multiple",
                        # 不在此处限制 file_types：Gradio 的类型校验依赖浏览器 MIME 识别，
                        # 在部分环境下会把合法的 .csv/.xlsx 等误判为非法类型。
                        # 文件扩展名的实际校验交由后端 utils/data_loader.py 处理（见 _on_data_upload）。
                    )
                    with gr.Row(elem_id="compose-toolbar"):
                        llm_dropdown = gr.Dropdown(
                            choices=llm_providers,
                            value=llm_providers[0],
                            label="模型",
                            scale=2,
                        )
                        token_slider = gr.Slider(
                            minimum=100000, maximum=10000000,
                            step=100000, value=1000000,
                            label="Token 预算",
                            scale=3,
                        )
                        generate_btn = gr.Button("✦ 生成分析计划", variant="primary",
                                                 elem_id="send-btn", scale=1)
                    use_search_checkbox = gr.Checkbox(
                        value=True,
                        label="联网检索（关闭后仅基于上传数据 / 问题描述分析；上传数据时会自动跳过检索）",
                    )

                # ---- 示例任务 ----
                with gr.Row():
                    example_btns = [gr.Button(t, size="sm", variant="secondary") for t in _EXAMPLE_TASKS]

                # ---- 高级设置（折叠） ----
                with gr.Accordion("⚙️ 高级设置（检索引擎 / LLM 接口配置）", open=False):
                    with gr.Row():
                        search_dropdown = gr.Dropdown(
                            choices=search_providers,
                            value=search_providers[0],
                            label="搜索提供商",
                        )
                        search_url_input = gr.Textbox(
                            label="检索引擎 URL 地址",
                            placeholder=_SEARCH_DEFAULT_URLS.get(search_providers[0], "检索引擎接口地址"),
                        )
                        search_key_input = gr.Textbox(
                            label="检索引擎 API Key",
                            placeholder="检索引擎的 API Key",
                            type="password",
                        )
                    with gr.Row():
                        llm_url_input = gr.Textbox(
                            label="LLM URL 地址",
                            placeholder="例如：https://api.openai.com/v1",
                        )
                        llm_key_input = gr.Textbox(
                            label="LLM API Key",
                            placeholder="LLM 服务的 API Key",
                            type="password",
                        )

                # ---- 计划展示 ----
                with gr.Group(visible=False) as plan_group:
                    gr.Markdown("### 分析计划预览", elem_classes="section-title")
                    plan_md = gr.Markdown()
                    selected_indices = gr.Textbox(
                        label="选择要执行的视角序号（如 1,3,5，留空则全部执行）",
                        placeholder="1,2,3",
                    )
                    custom_angles_input = gr.Textbox(
                        label="补充分析视角（可选，每行一条）",
                        placeholder="例如：\n各品牌市场份额对比\n价格区间分布及变化趋势",
                        lines=3,
                    )
                    team_size_slider = gr.Slider(
                        minimum=1, maximum=8, step=1, value=4,
                        label="分析团队规模（并行分析师数量）",
                    )
                    execute_btn = gr.Button("确认并执行分析", variant="primary")

                # ---- 执行进度 ----
                with gr.Group(visible=False) as progress_group:
                    gr.Markdown("### 执行进度", elem_classes="section-title")
                    progress_log = gr.Textbox(lines=10, interactive=False, show_label=False)

            # ==================== 右侧数据 / 报告面板 ====================
            with gr.Column(scale=3, elem_id="data-panel", min_width=320):
                gr.HTML('<div class="panel-title">📋 数据 / 报告</div>')
                data_preview_md = gr.Markdown(
                    value="> 上传数据后将在此预览；执行完成后将在此展示统计报告。",
                )

                with gr.Group(visible=False) as report_group:
                    gr.Markdown("### 统计报告", elem_classes="section-title")
                    report_preview = gr.Markdown(
                        latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                    )
                    with gr.Row():
                        download_btn = gr.Button("下载报告（含图表）", variant="secondary")
                        download_file = gr.File(label="报告压缩包（Markdown + 图表）", visible=False, interactive=False)

        # ==================== 事件处理 ====================

        # ---- 示例任务点击：填入输入框 ----
        for btn, txt in zip(example_btns, _EXAMPLE_TASKS):
            btn.click(fn=lambda t=txt: gr.update(value=t), inputs=None, outputs=[question_input])

        # ---- 新任务：重置界面 ----
        def on_new_task():
            return (
                gr.update(value=""),                       # question_input
                gr.update(value=None),                     # data_files_input
                gr.update(visible=False),                  # plan_group
                gr.update(visible=False),                  # progress_group
                gr.update(visible=False),                  # report_group
                gr.update(value="> 上传数据后将在此预览；执行完成后将在此展示统计报告。"),  # data_preview_md
                None,                                      # current_plan
                None,                                      # report_file
            )

        new_task_btn.click(
            fn=on_new_task,
            inputs=None,
            outputs=[question_input, data_files_input, plan_group, progress_group,
                     report_group, data_preview_md, current_plan, report_file],
        )

        # ---- 上传数据后实时解析并展示预览（右侧面板） ----
        def _on_data_upload(files):
            if not files:
                return (gr.update(value="> 上传数据后将在此预览；执行完成后将在此展示统计报告。"),
                        gr.update(value=True))
            from utils.data_loader import load_multiple
            paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
            loaded = load_multiple(paths)
            names = loaded.get("names", [])
            errors = loaded.get("errors", [])

            # 全部文件都解析失败：保持联网检索开关不变，明确提示失败原因。
            if not names:
                err_md = "\n".join(f"- {e}" for e in errors) or "- 未知错误"
                md = (
                    "#### ❌ 数据加载失败\n\n"
                    f"{err_md}\n\n"
                    "> 仅支持 `.csv / .tsv / .xlsx / .xls / .json / .txt / .md / .markdown`，"
                    "请确认文件内容与扩展名一致。"
                )
                return gr.update(value=md), gr.update()

            names_str = "、".join(names)
            preview = loaded.get("preview", "")
            md = f"#### 已上传数据：{names_str}\n\n{preview}\n\n> 执行分析时将基于以上数据，**跳过联网检索**。"
            # 上传数据成功后自动取消联网检索（分析将完全基于上传数据）
            return gr.update(value=md), gr.update(value=False)

        data_files_input.change(
            fn=_on_data_upload,
            inputs=[data_files_input],
            outputs=[data_preview_md, use_search_checkbox],
        )

        # ---- 切换搜索提供商时更新占位提示 ----
        def _on_search_provider_change(provider):
            return gr.update(placeholder=_SEARCH_DEFAULT_URLS.get(provider, "检索引擎接口地址"))

        search_dropdown.change(
            fn=_on_search_provider_change,
            inputs=[search_dropdown],
            outputs=[search_url_input],
        )

        # ---- 生成分析计划 ----
        async def on_generate(question, data_files, mode, search_p, search_url, search_key, llm_p, llm_url, llm_key, budget):
            """生成分析计划（仅生成计划，不触发搜索）。"""
            data_text = None
            if data_files:
                try:
                    from utils.data_loader import load_multiple
                    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in data_files]
                    loaded = load_multiple(paths)
                    data_text = loaded.get("text") or None
                except Exception as e:
                    yield f"数据加载失败: {e}", None, gr.update(visible=False), gr.update(visible=False)
                    return

            if (not question or not question.strip()) and not data_text:
                yield "请输入问题，或上传自定义数据", None, gr.update(visible=False), gr.update(visible=False)
                return

            yield "⏳ 正在生成分析计划，请稍候……", None, gr.update(visible=True), gr.update(visible=False)

            _apply_runtime_config(search_p, search_url, search_key, llm_p, llm_url, llm_key)

            agent = StatisticAgent(token_budget=budget)

            q = (question or "基于上传数据的统计分析").strip()
            task = asyncio.create_task(agent.generate_plan(q, data_text=data_text, mode=mode or "report"))
            start = time.time()
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=0.5)
                if task in done:
                    break
                elapsed = int(time.time() - start)
                yield (f"⏳ 正在生成分析计划，请稍候……（已用时 {elapsed}s）",
                       None, gr.update(visible=True), gr.update(visible=False))

            try:
                plan = task.result()

                data_source = "用户上传数据（跳过联网检索）" if plan.data_text else "联网检索"
                md = f"""## 分析计划

> **问题**: {plan.question}
> **数据来源**: {data_source}
> **预计图表数**: {plan.estimated_charts}
> **预计 Token**: {plan.estimated_tokens}

### 分析视角列表

"""
                for idx, angle in enumerate(plan.angles):
                    if isinstance(angle, dict):
                        md += f"""**{idx + 1}. {angle.get('insight', '')[:60]}**

- 洞察: {angle.get('insight', '')}
- 计划: {angle.get('plan', '')}
- 图表: {'是' if angle.get('need_plot') else '否'} — {angle.get('plot_title', '无')}

---

"""

                yield md, plan, gr.update(visible=True), gr.update(visible=False)

            except Exception as e:
                yield f"生成计划失败: {e}", None, gr.update(visible=False), gr.update(visible=False)

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input, data_files_input, mode_radio, search_dropdown, search_url_input, search_key_input,
                    llm_dropdown, llm_url_input, llm_key_input, token_slider],
            outputs=[plan_md, current_plan, plan_group, report_group],
        )

        # ---- 执行分析计划 ----
        async def on_execute(indices_text, custom_angles, team_size, use_search, data_files, plan, search_p, search_url, search_key, llm_p, llm_url, llm_key):
            """执行分析计划（用户确认 / 编辑 / 补充后，在此阶段才进行搜索等操作）。"""
            if not plan:
                yield "请先生成分析计划", "", gr.update(visible=False), gr.update(visible=False), None
                return

            _apply_runtime_config(search_p, search_url, search_key, llm_p, llm_url, llm_key)

            # 执行前重新读取当前上传的文件，确保优先使用最新的（可能多个）上传数据，
            # 而不是一味联网检索；也覆盖"生成计划后又增删文件"的情况。
            if data_files:
                try:
                    from utils.data_loader import load_multiple
                    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in data_files]
                    loaded = load_multiple(paths)
                    latest_text = loaded.get("text") or None
                    if latest_text and latest_text.strip():
                        plan.data_text = latest_text
                except Exception:
                    # 重新加载失败时，回退到计划中已捕获的 data_text（若有）
                    pass

            agent = StatisticAgent(token_budget=plan.estimated_tokens, team_size=int(team_size or 4))

            confirmed = None
            if indices_text and indices_text.strip():
                try:
                    idxs = [int(x.strip()) - 1 for x in indices_text.split(",")]
                    confirmed = []
                    for i in idxs:
                        if 0 <= i < len(plan.steps):
                            confirmed.append(plan.steps[i].step_id)
                except Exception:
                    pass

            custom_step_ids = agent.add_custom_angles(plan, custom_angles or "")
            if custom_step_ids and confirmed is not None:
                confirmed.extend(custom_step_ids)

            logs: List[str] = ["开始执行分析计划……"]

            queue: "asyncio.Queue[str]" = asyncio.Queue()

            def on_progress(p):
                try:
                    queue.put_nowait(
                        f"[{p.current_step}/{p.total_steps}] {p.step_type}: {p.step_description} ({p.status})"
                    )
                except Exception:
                    pass

            agent.on_progress(on_progress)

            yield "\n".join(logs), "", gr.update(visible=True), gr.update(visible=False), None

            task = asyncio.create_task(agent.execute_plan(plan, confirmed_steps=confirmed, use_search=bool(use_search)))
            start = time.time()
            while not task.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                    logs.append(msg)
                    yield "\n".join(logs), "", gr.update(visible=True), gr.update(visible=False), None
                except asyncio.TimeoutError:
                    elapsed = int(time.time() - start)
                    heartbeat = "\n".join(logs + [f"  …运行中（已用时 {elapsed}s）"])
                    yield heartbeat, "", gr.update(visible=True), gr.update(visible=False), None

            while not queue.empty():
                try:
                    logs.append(queue.get_nowait())
                except Exception:
                    break

            try:
                result = task.result()
                logs.append("执行完成！")
                report_md = result.get("report_md", "")
                rpath = result.get("report_path", "")
                # 将报告中引用的本地图片转为 base64 内嵌，确保在前端 Markdown 中正常显示
                report_preview_md = _embed_images_base64(report_md)
                yield "\n".join(logs), report_preview_md, gr.update(visible=True), gr.update(visible=True), rpath
            except Exception as e:
                logs.append(f"执行失败: {e}")
                yield "\n".join(logs), f"发生错误: {e}", gr.update(visible=True), gr.update(visible=False), None

        execute_btn.click(
            fn=on_execute,
            inputs=[selected_indices, custom_angles_input, team_size_slider, use_search_checkbox, data_files_input, current_plan,
                    search_dropdown, search_url_input, search_key_input,
                    llm_dropdown, llm_url_input, llm_key_input],
            outputs=[progress_log, report_preview, progress_group, report_group, report_file],
        )

        def on_download(rpath):
            if rpath and os.path.exists(rpath):
                # 打包报告及其引用的图片为 zip，确保下载内容附带图表
                zip_path = _make_report_zip(rpath)
                target = zip_path or rpath
                return gr.update(value=target, visible=True)
            return gr.update(value=None, visible=False)

        download_btn.click(
            fn=on_download,
            inputs=[report_file],
            outputs=[download_file],
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    # 启用事件队列，确保生成器式事件能够实时把进度流式推送到前端（避免假死）
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
