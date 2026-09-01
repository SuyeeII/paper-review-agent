"""论文多视角审稿助手 - Gradio 界面（多Agent并行评审架构）
运行：python app.py
然后浏览器打开 http://localhost:7860
"""
import os
import re
import tempfile
import gradio as gr
from debate_agent import run_review, format_review_result
from debate_agent.state import ReviewState

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


def generate_radar_chart(scores: dict) -> str:
    """
    生成4个维度评分的雷达图
    scores: {"创新性": 7, "方法论": 6, "论证与证据": 7, "写作表达": 7}
    返回雷达图的临时文件路径，如果生成失败返回None
    """
    if not HAS_MATPLOTLIB:
        return None

    try:
        labels = list(scores.keys())
        values = list(scores.values())

        # 雷达图需要闭合
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color='#4A90D9')
        ax.fill(angles, values, alpha=0.25, color='#4A90D9')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        # 设置y轴范围
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)

        # 在每个点上显示分数
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + 0.5, str(value), ha='center', va='center', fontsize=11, fontweight='bold')

        plt.title('论文审稿评分雷达图', fontsize=14, pad=20)
        plt.tight_layout()

        # 保存为临时文件
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp_file.name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        return tmp_file.name

    except Exception as e:
        print(f"[雷达图] 生成失败: {e}")
        return None


def generate_html_report(markdown_content: str, title: str = "论文审稿报告") -> str:
    """
    生成美观的HTML审稿报告
    用户可以用浏览器打开后打印为PDF
    返回HTML文件的临时路径
    """
    try:
        # 简单的Markdown转HTML（处理基本格式）
        html_content = markdown_content
        # 处理标题
        html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
        # 处理加粗
        html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        # 处理表格（简单处理）
        html_content = re.sub(r'\|(.+)\|', r'<tr><td>\1</td></tr>', html_content)
        html_content = html_content.replace('|', '</td><td>')
        # 处理列表
        html_content = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
        # 处理换行
        html_content = html_content.replace('\n', '<br>\n')

        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "SimHei", sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.8;
            color: #333;
            background-color: #fff;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h2 {{
            color: #2980b9;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 25px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        td, th {{
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        li {{
            margin: 8px 0;
        }}
        strong {{
            color: #e74c3c;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .header h1 {{
            color: white;
            border: none;
            margin: 0;
        }}
        @media print {{
            body {{
                padding: 20px;
            }}
            .header {{
                background: #667eea !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📝 {title}</h1>
        <p>论文多视角审稿助手 · 自动生成</p>
    </div>
    {html_content}
    <hr>
    <p style="text-align: center; color: #999; font-size: 12px;">
        本报告由论文多视角审稿助手自动生成，仅供参考，最终审稿决定请以期刊/会议官方意见为准。
    </p>
</body>
</html>"""

        tmp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
        tmp_file.write(html_template)
        tmp_file.close()
        return tmp_file.name

    except Exception as e:
        print(f"[HTML导出] 生成失败: {e}")
        return None


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


def on_paper_file_upload(file_obj) -> str:
    """
    文件上传后，解析论文内容并返回
    内容存在后台State里，不显示在界面上
    """
    parsed = parse_paper_file(file_obj)
    return parsed if parsed else ""


def run_review_ui(paper_content: str):
    """
    Gradio 界面的审稿运行函数
    4个维度审稿人并行评审 → 主编汇总
    返回：论文结构解析、4个维度审稿意见、主编报告、导出文件、状态
    """
    if not paper_content or not paper_content.strip():
        yield "请先上传论文文件（PDF / TXT / MD）！", "", "", "", "", "", "", None, "❌ 未上传论文文件，请先上传", None, None
        return

    # 先yield一次，显示"正在审稿"提示（不带百分比，因为同步执行无法实时更新进度）
    yield "", "", "", "", "", "", "", None, "正在审稿，请稍等几分钟...", None, None

    # 运行审稿（同步执行，因为LangGraph invoke是同步的）
    try:
        final_state = run_review(
            topic=paper_content.strip(),
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
        yield friendly_msg, "", "", "", "", "", "", None, "", None, None
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

    # 导出为HTML文件（用户可以用浏览器打开后打印为PDF）
    html_export_path = generate_html_report(full_text, "论文多视角审稿报告")
    if html_export_path:
        print(f"[HTML导出] 已保存到: {html_export_path}")

    # 从主编报告中提取4个维度的得分，生成雷达图
    radar_chart_path = None
    try:
        # 用正则表达式从评分表格中提取得分
        score_pattern = r'\|\s*(创新性|方法论|论证与证据|写作表达)\s*\|\s*(\d+)\s*/\s*10\s*\|'
        matches = re.findall(score_pattern, summary_text)
        if matches:
            scores = {dim: int(score) for dim, score in matches}
            radar_chart_path = generate_radar_chart(scores)
            print(f"[雷达图] 生成成功，得分: {scores}")
    except Exception as e:
        print(f"[雷达图] 生成失败: {e}")

    yield structure_text, innovation_text, methodology_text, experiment_text, writing_text, summary_text, full_text, export_path, "✅ 审稿完成！", radar_chart_path, html_export_path


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
        # 右侧：主编综合报告 + 雷达图
        with gr.Column(scale=2):
            radar_image = gr.Image(label="📊 评分雷达图", interactive=False, height=350)
            summary_md = gr.Markdown("审稿结果会在审稿结束后展示...", label="📊 主编综合审稿报告")

    # 下方：完整报告复制框 + 导出下载
    full_text_box = gr.Textbox(
        label="📋 完整审稿报告（可全选复制）",
        lines=20,
        interactive=False,
    )
    with gr.Row():
        export_file = gr.File(label="📥 导出 Markdown", interactive=False)
        html_export_file = gr.File(label="📥 导出 HTML（可浏览器打开后打印为PDF）", interactive=False)

    # 事件绑定
    run_button.click(
        fn=run_review_ui,
        inputs=[paper_content_state],
        outputs=[structure_md, innovation_md, methodology_md, experiment_md, writing_md,
                 summary_md, full_text_box, export_file, status_text, radar_image, html_export_file],
    )

    # 页脚
    gr.Markdown("""
    ---
    <sub>**技术栈**：LangGraph（多Agent并行）· LangChain · OpenAI 兼容 API（智谱 GLM）· Gradio · PyPDF2</sub>
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())



