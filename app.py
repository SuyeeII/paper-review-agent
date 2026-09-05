"""论文多视角审稿助手 - Gradio 界面（多Agent并行评审架构）
运行：python app.py
然后浏览器打开 http://localhost:7860
"""
import os
import re
import tempfile
import gradio as gr
from paper_review_agent import run_review, format_review_result
from paper_review_agent.state import ReviewState
from paper_review_agent.pdf_parser import parse_paper_file

# 全局：最近一次上传文件的图表信息（tables_md/images/tables），审稿时透传
_last_parse_meta = {"tables_md": "", "images": 0, "tables": 0}


# 全局：最近一次上传文件的图表信息（tables_md/images/tables），审稿时透传
_last_parse_meta = {"tables_md": "", "images": 0, "tables": 0}
# 全局：最近一次解析的错误/警告信息（供界面显示具体原因）
_last_parse_error = ""


def parse_paper_file_ui(file_obj) -> str:
    """
    解析上传的论文文件，提取文本内容
    支持 PDF / TXT / MD
    返回提取的文本内容，如果解析失败返回空字符串
    解析失败的具体原因记录到 _last_parse_error，供界面显示
    """
    global _last_parse_error
    _last_parse_error = ""

    if file_obj is None:
        _last_parse_error = "未检测到上传文件"
        return ""

    # Gradio 的 File 组件返回的是文件路径字符串（单文件）或列表
    file_path = file_obj if isinstance(file_obj, str) else (file_obj[0] if isinstance(file_obj, list) and file_obj else None)
    if not file_path or not os.path.exists(file_path):
        _last_parse_error = f"文件路径不存在：{file_path}"
        return ""

    try:
        parsed = parse_paper_file(file_path)
        # 保存图表信息供审稿透传
        _last_parse_meta["tables_md"] = parsed.get("tables_md", "")
        _last_parse_meta["images"] = parsed.get("images", 0)
        _last_parse_meta["tables"] = parsed.get("tables", 0)
        if parsed.get("parser") == "pdfplumber":
            print(f"[PDF解析] pdfplumber 提取文本 {len(parsed['text'])} 字符，表格 {parsed['tables']} 个，图片 {parsed['images']} 幅")
        if parsed.get("warning"):
            print(f"[PDF解析提示] {parsed['warning']}")
            _last_parse_error = parsed["warning"]
        text = parsed.get("text", "")
        if not text.strip():
            # 无文本（扫描件/图片型PDF等）：给用户明确提示
            if not _last_parse_error:
                _last_parse_error = "PDF 中未提取到任何文字，可能是扫描件/图片型 PDF（没有文字层）"
            return ""
        return text
    except Exception as e:
        _last_parse_error = f"文件解析异常：{e}"
        print(f"[文件解析失败] {e}")
        return ""


def on_paper_file_upload(file_obj) -> str:
    """
    文件上传后，解析论文内容并返回
    内容存在后台State里，不显示在界面上
    解析失败时返回空字符串（不把错误提示当论文内容），并在控制台打印原因
    """
    parsed = parse_paper_file_ui(file_obj)
    if not parsed:
        return ""
    if parsed.startswith("⚠️"):
        # 解析失败：打印诊断信息，返回空，避免错误提示被当作论文内容传入审稿流程
        print(f"[文件解析失败] {parsed}")
        return ""
    return parsed


def run_review_ui(paper_content: str):
    """
    Gradio 界面的审稿运行函数
    4个维度审稿人并行评审 → 主编汇总
    返回：论文结构解析、4个维度审稿意见、主编报告、导出文件、状态
    """
    if not paper_content or not paper_content.strip():
        reason = _last_parse_error if _last_parse_error else "未上传文件或文件解析失败"
        yield f"❌ 未获取到论文内容。原因：{reason}。若是扫描件/图片型 PDF，请先转成可复制文字的 PDF（或用 OCR 工具提取文本）后再上传。", "", "", "", "", "", "", None, "❌ 未获取到论文内容"
        return

    # 先yield一次，显示"正在审稿"提示（不带百分比，因为同步执行无法实时更新进度）
    yield "", "", "", "", "", "", "", None, "正在审稿，请稍等几分钟..."

    # 运行审稿（同步执行，因为LangGraph invoke是同步的）
    try:
        final_state = run_review(
            topic=paper_content.strip(),
            tables_md=_last_parse_meta.get("tables_md", ""),
            images=_last_parse_meta.get("images", 0),
            tables=_last_parse_meta.get("tables", 0),
        )
    except Exception as e:
        error_msg = str(e)
        # 根据错误类型给出更友好的提示和解决建议
        if "api_key" in error_msg.lower() or "API Key" in error_msg or "未找到 API Key" in error_msg:
            friendly_msg = "❌ 未配置 API Key。请在项目根目录创建 .env 文件，设置 ZHIPU_API_KEY（或 OPENAI_API_KEY）、ZHIPU_BASE_URL、ZHIPU_MODEL。"
        elif "timeout" in error_msg.lower() or "超时" in error_msg or "timed out" in error_msg.lower():
            friendly_msg = "❌ 模型调用超时。可能是网络不稳定或模型响应慢，请稍后重试，或检查网络连接。"
        elif "rate limit" in error_msg.lower() or "限流" in error_msg or "429" in error_msg:
            friendly_msg = "❌ API 调用频率超限。请稍后再试，或检查 API 配额。"
        elif "connection" in error_msg.lower() or "连接" in error_msg or "网络" in error_msg:
            friendly_msg = "❌ 网络连接失败。请检查网络连接，或确认 API 地址（ZHIPU_BASE_URL）是否正确。"
        elif "invalid" in error_msg.lower() and ("key" in error_msg.lower() or "api" in error_msg.lower()):
            friendly_msg = "❌ API Key 无效。请检查 .env 文件中的 ZHIPU_API_KEY 是否正确。"
        else:
            friendly_msg = f"❌ 审稿运行出错：{error_msg[:200]}"
        yield friendly_msg, "", "", "", "", "", "", None, ""
        return

    # 提取各部分内容
    structure_text = final_state.get("paper_structure") or "论文结构解析失败"
    innovation_text = final_state.get("innovation_review") or "创新性审稿失败"
    methodology_text = final_state.get("methodology_review") or "方法论审稿失败"
    experiment_text = final_state.get("experiment_review") or "论证与证据审稿失败"
    writing_text = final_state.get("writing_review") or "写作审稿失败"
    summary_text = final_state.get("editor_summary") or "主编汇总生成失败"

    # 完整文本（用于导出文件和底部复制框）
    full_text = format_review_result(final_state)

    # 导出为markdown文件
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"审稿报告_{timestamp}.md")
    try:
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"[导出] 审稿报告已保存到: {export_path}")
    except Exception as e:
        print(f"[导出] 保存失败: {e}")
        export_path = None

    yield structure_text, innovation_text, methodology_text, experiment_text, writing_text, summary_text, full_text, export_path, "✅ 审稿完成！"


# ===== Gradio 界面 =====

with gr.Blocks(title="论文多视角审稿助手") as demo:
    gr.Markdown("""
    # 📝 论文多视角审稿助手

    基于 LangGraph 的多智能体论文审稿系统，4个维度审稿人（创新性/方法论/论证与证据/写作表达）并行评审，由主编汇总，输出结构化综合审稿报告。

    **审稿维度**：创新性 · 方法论 · 论证与证据 · 写作表达

    > 💡 请上传完整论文 PDF（或 TXT/MD），系统会自动解析全文进行审稿。
    """)

    # ===== 输入区（只保留文件上传，去掉手动输入框）=====
    paper_content_state = gr.State("")  # 后台保存解析后的论文文本
    paper_file_input = gr.File(
        label="📄 上传论文文件（PDF / TXT / MD，必填）",
        file_count="single",
        file_types=[".pdf", ".txt", ".md"],
    )
    paper_file_input.change(
        fn=on_paper_file_upload,
        inputs=[paper_file_input],
        outputs=[paper_content_state],
    )

    # 开始审稿按钮占满整行，状态显示在按钮下方
    run_button = gr.Button("🚀 开始审稿", variant="primary", size="lg")
    status_text = gr.Textbox(label="状态", value="等待开始...", interactive=False, lines=1)

    # ===== 输出区 =====
    with gr.Row():
        # 左侧：分维度Tab展示
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("📄 论文结构解析"):
                    structure_md = gr.Markdown("审稿开始后这里会展示论文结构解析结果...")
                with gr.Tab("💡 创新性"):
                    innovation_md = gr.Markdown("审稿开始后这里会展示创新性审稿意见...")
                with gr.Tab("🔧 方法论"):
                    methodology_md = gr.Markdown("审稿开始后这里会展示方法论审稿意见...")
                with gr.Tab("⚖️ 论证与证据"):
                    experiment_md = gr.Markdown("审稿开始后这里会展示论证与证据审稿意见...")
                with gr.Tab("✍️ 写作表达"):
                    writing_md = gr.Markdown("审稿开始后这里会展示写作审稿意见...")
        # 右侧：主编综合报告
        with gr.Column(scale=2):
            summary_md = gr.Markdown("审稿结果会在审稿结束后展示...", label="📊 主编综合审稿报告")

    # 下方：完整报告复制框 + 导出下载
    full_text_box = gr.Textbox(
        label="📋 完整审稿报告（可全选复制）",
        lines=20,
        interactive=False,
    )
    with gr.Row():
        export_file = gr.File(label="📥 导出审稿报告（Markdown，审稿完成后可下载）", interactive=False)

    # 事件绑定
    run_button.click(
        fn=run_review_ui,
        inputs=[paper_content_state],
        outputs=[structure_md, innovation_md, methodology_md, experiment_md, writing_md,
                 summary_md, full_text_box, export_file, status_text],
    )

    # 页脚
    gr.Markdown("""
    ---
    <sub>**技术栈**：LangGraph（多Agent并行）· LangChain · OpenAI 兼容 API（智谱 GLM）· Gradio · pdfplumber/PyPDF2</sub>
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())



