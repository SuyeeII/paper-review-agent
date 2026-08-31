"""论文多视角审稿助手 - Gradio 界面（多Agent并行评审架构）
运行：python app.py
然后浏览器打开 http://localhost:7860
"""
import os
import gradio as gr
from debate_agent import run_review, format_review_result
from debate_agent.state import ReviewState

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


def parse_paper_file(file_obj) -> str:
    """
    解析上传的论文文件，提取文本内容
    支持 PDF / TXT / MD
    返回提取的文本内容，如果解析失败返回空字符串
    """
    if file_obj is None:
        return ""

    # Gradio 的 File 组件返回的是文件路径字符串（单文件）或列表
    file_path = file_obj if isinstance(file_obj, str) else (file_obj[0] if isinstance(file_obj, list) and file_obj else None)
    if not file_path or not os.path.exists(file_path):
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            if not HAS_PYPDF2:
                return "⚠️ PDF 解析库未安装，请安装 PyPDF2 后重试，或直接粘贴论文文本。"
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)

        elif ext in (".txt", ".md", ".markdown"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        else:
            return f"⚠️ 不支持的文件格式：{ext}，请上传 PDF / TXT / MD 文件，或直接粘贴论文文本。"

    except Exception as e:
        return f"⚠️ 文件解析失败：{str(e)}，请尝试直接粘贴论文文本。"


def on_paper_file_upload(file_obj, current_text: str) -> str:
    """
    文件上传后，把解析的内容替换文本框内容
    直接替换，避免和已有内容混在一起导致审稿对象混乱
    """
    parsed = parse_paper_file(file_obj)
    if not parsed:
        return current_text
    # 直接替换文本框内容，不追加
    return parsed


def run_review_ui(topic: str, reflection_enabled: bool, progress=gr.Progress()):
    """
    Gradio 界面的审稿运行函数
    4个维度审稿人并行评审 → 自我反思修正 → 主编汇总
    返回：论文结构解析、4个维度审稿意见、主编报告、完整记录、导出文件、状态
    """
    if not topic or not topic.strip():
        yield "请输入论文内容，或上传论文文件！", "", "", "", "", "", "", None, ""
        return

    # 阶段进度映射
    stages = [
        (0.05, "解析论文结构中..."),
        (0.15, "创新性审稿人评审中..."),
        (0.3, "方法论审稿人评审中..."),
        (0.45, "实验审稿人评审中..."),
        (0.6, "写作审稿人评审中..."),
    ]
    if reflection_enabled:
        stages.extend([
            (0.7, "创新性审稿人自我反思修正中..."),
            (0.75, "方法论审稿人自我反思修正中..."),
            (0.8, "实验审稿人自我反思修正中..."),
            (0.85, "写作审稿人自我反思修正中..."),
        ])
    stages.append((0.95, "主编汇总综合审稿报告中..."))

    # 显示初始进度
    for p, desc in stages:
        progress(p, desc=desc)

    # 运行审稿（同步执行，因为LangGraph invoke是同步的）
    try:
        final_state = run_review(
            topic=topic.strip(),
            reflection_enabled=reflection_enabled,
        )
    except Exception as e:
        error_msg = f"❌ 审稿运行出错：{str(e)}"
        yield error_msg, "", "", "", "", "", "", None, ""
        return

    # 提取各部分内容
    structure_text = final_state.get("paper_structure") or "论文结构解析失败"
    innovation_text = final_state.get("innovation_final") or final_state.get("innovation_review") or "创新性审稿失败"
    methodology_text = final_state.get("methodology_final") or final_state.get("methodology_review") or "方法论审稿失败"
    experiment_text = final_state.get("experiment_final") or final_state.get("experiment_review") or "实验审稿失败"
    writing_text = final_state.get("writing_final") or final_state.get("writing_review") or "写作审稿失败"
    summary_text = final_state.get("editor_summary") or "主编汇总生成失败"

    # 完整文本
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

    progress(1.0, desc="✅ 审稿完成！")
    yield structure_text, innovation_text, methodology_text, experiment_text, writing_text, summary_text, full_text, export_path, "✅ 审稿完成！"


# ===== Gradio 界面 =====

with gr.Blocks(title="论文多视角审稿助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 论文多视角审稿助手

    基于 LangGraph 的多智能体论文审稿系统，4个维度审稿人（创新性/方法论/实验/写作）并行评审，自我反思修正后由主编汇总，输出结构化综合审稿报告。

    **审稿维度**：创新性 · 方法论 · 实验可靠性与可复现性 · 写作表达

    > 💡 建议上传完整论文 PDF，或直接粘贴论文全文，审稿效果更好。
    """)

    # ===== 输入区 =====
    topic_input = gr.Textbox(
        label="论文标题/摘要/全文",
        placeholder="粘贴论文标题、摘要或全文，或上传下方论文文件...",
        lines=6,
        value="",
    )
    paper_file_input = gr.File(
        label="📄 上传论文文件（PDF / TXT / MD，上传后自动解析填入上方文本框）",
        file_count="single",
        file_types=[".pdf", ".txt", ".md"],
    )
    paper_file_input.change(
        fn=on_paper_file_upload,
        inputs=[paper_file_input, topic_input],
        outputs=[topic_input],
    )

    with gr.Row():
        reflection_checkbox = gr.Checkbox(
            value=True,
            label="自我反思修正（推荐）",
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
                with gr.Tab("🧪 实验与可复现性"):
                    experiment_md = gr.Markdown("审稿开始后这里会展示实验审稿意见...")
                with gr.Tab("✍️ 写作表达"):
                    writing_md = gr.Markdown("审稿开始后这里会展示写作审稿意见...")
        # 右侧：主编综合报告
        with gr.Column(scale=2):
            summary_md = gr.Markdown("审稿结果会在审稿结束后展示...", label="📊 主编综合审稿报告")

    # 下方：完整记录 + 导出下载
    with gr.Row():
        with gr.Column(scale=3):
            full_text_box = gr.Textbox(label="完整审稿记录（可全选复制）", lines=6, interactive=False)
        with gr.Column(scale=1):
            export_file = gr.File(label="📥 导出审稿报告（Markdown）", interactive=False)

    # 事件绑定
    run_button.click(
        fn=run_review_ui,
        inputs=[topic_input, reflection_checkbox],
        outputs=[structure_md, innovation_md, methodology_md, experiment_md, writing_md,
                 summary_md, full_text_box, export_file, status_text],
    )

    # 页脚
    gr.Markdown("""
    ---
    <sub>**技术栈**：LangGraph（多Agent并行）· LangChain · OpenAI 兼容 API（智谱 GLM）· Gradio · PyPDF2</sub>
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
