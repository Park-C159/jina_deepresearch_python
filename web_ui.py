import asyncio
import os

import gradio as gr

from agent import get_response
from core.plugin_manager import get_manager


def _get_search_providers():
    """从插件注册表动态获取所有搜索提供商"""
    try:
        mgr = get_manager()
        plugins = mgr.list("search")
        return list(plugins.keys()) if plugins else ["jina", "milvus"]
    except Exception:
        return ["jina", "milvus"]


def _get_llm_providers():
    """从插件注册表动态获取所有 LLM 提供商"""
    try:
        mgr = get_manager()
        plugins = mgr.list("llm")
        return list(plugins.keys()) if plugins else ["openai", "gemini", "qwen"]
    except Exception:
        return ["openai", "gemini", "qwen"]


async def research(
    question: str,
    search_provider: str,
    llm_provider: str,
    token_budget: int,
    max_bad_attempts: int,
):
    if not question or not question.strip():
        return "请输入问题", ""

    # 动态切换 LLM 提供商（通过环境变量）
    os.environ["LLM_PROVIDER"] = llm_provider

    status_lines = [f"开始研究: {question.strip()}", f"搜索提供商: {search_provider}", f"LLM 提供商: {llm_provider}"]

    try:
        result = await get_response(
            question=question.strip(),
            search_languge_code="zh",
            search_provider=search_provider if search_provider != "none" else None,
            language_code="zh",
            with_images=False,
            token_budget=token_budget,
            max_bad_attempts=max_bad_attempts,
            existing_context=None,
            messages=[],
            num_returned_urls=10,
            no_direct_answer=False,
            max_ref=10,
            min_rel_score=0.7,
            team_size=1,
        )

        answer_data = result.get("result", {})
        md_answer = answer_data.get("mdAnswer", answer_data.get("answer", "无答案"))
        read_urls = result.get("readURLs", [])

        status_lines.append("研究完成")

        refs = "\n".join(f"- {url}" for url in read_urls) if read_urls else "无"
        status_text = "\n".join(status_lines)
        full_answer = f"{md_answer}\n\n---\n\n**参考来源**:\n{refs}"
        return status_text, full_answer

    except Exception as e:
        status_lines.append(f"错误: {e}")
        return "\n".join(status_lines), f"发生错误: {e}"


def create_ui():
    search_providers = _get_search_providers()
    llm_providers = _get_llm_providers()

    with gr.Blocks(title="Deep Research Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Deep Research Assistant")
        gr.Markdown("输入问题，AI 将自动进行多轮搜索、阅读与反思，生成深度研究报告。")

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="研究问题",
                    placeholder="例如：2025年人工智能在医疗领域的最新进展有哪些？",
                    lines=3,
                )
                with gr.Row():
                    provider_dropdown = gr.Dropdown(
                        choices=search_providers,
                        value=search_providers[0] if search_providers else "jina",
                        label="搜索提供商",
                    )
                    llm_dropdown = gr.Dropdown(
                        choices=llm_providers,
                        value=llm_providers[0] if llm_providers else "openai",
                        label="LLM 提供商",
                    )
                with gr.Row():
                    token_slider = gr.Slider(
                        minimum=100000,
                        maximum=10000000,
                        step=100000,
                        value=1000000,
                        label="Token 预算",
                    )
                    attempts_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=2,
                        label="最大重试次数",
                    )
                submit_btn = gr.Button("开始研究", variant="primary")

                with gr.Accordion("已注册插件", open=False):
                    plugin_info = gr.JSON(
                        value=get_manager().list(),
                        label="插件列表",
                    )

            with gr.Column(scale=3):
                status_output = gr.Textbox(
                    label="运行状态",
                    lines=5,
                    interactive=False,
                )
                answer_output = gr.Markdown(
                    label="研究报告",
                    latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
                )

        submit_btn.click(
            fn=research,
            inputs=[
                question_input,
                provider_dropdown,
                llm_dropdown,
                token_slider,
                attempts_slider,
            ],
            outputs=[status_output, answer_output],
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
