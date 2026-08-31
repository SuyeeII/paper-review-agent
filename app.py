"""多 Agent 辩论系统 - Gradio 界面
运行：python app.py
然后浏览器打开 http://localhost:7860
"""
import gradio as gr
from debate_agent import run_debate, format_debate_result
from debate_agent.state import DebateState


def run_debate_ui(topic: str, max_rounds: int, reflection_enabled: bool,
                  affirmative_stance: str, negative_stance: str,
                  rag_enabled: bool, aff_evidence_files, neg_evidence_files,
                  web_search_enabled: bool, progress=gr.Progress()):
    """
    Gradio 界面的辩论运行函数
    使用 progress 实时展示辩论进度
    """
    if not topic or not topic.strip():
        yield "请输入辩题！", "", "", ""
        return

    # 如果用户没填立场表述，用默认值
    aff_stance = affirmative_stance.strip() if affirmative_stance and affirmative_stance.strip() else None
    neg_stance = negative_stance.strip() if negative_stance and negative_stance.strip() else None

    # V2 RAG：处理上传的论据文件
    aff_files = []
    neg_files = []
    if rag_enabled:
        # gr.File 上传多个文件时返回路径列表，单个文件返回字符串
        if aff_evidence_files:
            if isinstance(aff_evidence_files, list):
                aff_files = [f if isinstance(f, str) else f.name for f in aff_evidence_files]
            else:
                aff_files = [aff_evidence_files if isinstance(aff_evidence_files, str) else aff_evidence_files.name]
        if neg_evidence_files:
            if isinstance(neg_evidence_files, list):
                neg_files = [f if isinstance(f, str) else f.name for f in neg_evidence_files]
            else:
                neg_files = [neg_evidence_files if isinstance(neg_evidence_files, str) else neg_evidence_files.name]
        print(f"[V2 RAG] 正方论据文件: {aff_files}")
        print(f"[V2 RAG] 反方论据文件: {neg_files}")

    transcript_lines = []
    final_state = None

    def progress_callback(node_name: str, state: DebateState):
        """每个节点执行完后的回调，用于实时更新界面"""
        nonlocal final_state
        final_state = state

        # V1.5：打印步骤日志，方便定位卡在哪一步
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 节点完成: {node_name} | 当前轮次: {state.get('current_round', 0)} | 阶段: {state.get('phase', '')}")

        # 找到最新的发言
        if state["full_transcript"]:
            latest = state["full_transcript"][-1]
            role = latest["role"]
            content = latest["content"]
            transcript_lines.append(f"### 【{role}】\n\n{content}\n\n---\n")

        # 阶段名称映射
        stage_names = {
            "opening_affirmative": "支持接受方立论中...",
            "opening_negative": "建议拒稿方立论中...",
            "clash_affirmative": f"支持接受方第{state.get('current_round', 1)}轮审稿中...",
            "clash_negative": f"建议拒稿方第{state.get('current_round', 1)}轮审稿中...",
            "reflection": "双方自我反思中...",
            "summary": "记忆摘要压缩中...",
            "rebuttal_affirmative": "支持接受方反驳中...",
            "rebuttal_negative": "建议拒稿方反驳中...",
            "closing_affirmative": "支持接受方总结中...",
            "closing_negative": "建议拒稿方总结中...",
            "judge": "综合审稿意见生成中...",
        }
        progress(0.5, desc=stage_names.get(node_name, "审稿进行中..."))

    # 运行审稿
    try:
        progress(0.1, desc="初始化论文审稿...")
        final_state = run_debate(
            topic=topic.strip(),
            max_rounds=int(max_rounds),
            reflection_enabled=reflection_enabled,
            affirmative_stance=aff_stance,
            negative_stance=neg_stance,
            progress_callback=progress_callback,
            rag_enabled=rag_enabled,
            affirmative_evidence_files=aff_files if rag_enabled else None,
            negative_evidence_files=neg_files if rag_enabled else None,
            web_search_enabled=web_search_enabled,
        )
    except Exception as e:
        error_msg = f"❌ 分析运行出错：{str(e)}"
        yield error_msg, "", "", ""
        return

    # 整理完整 transcript
    full_transcript = ""
    for item in final_state["full_transcript"]:
        full_transcript += f"### 【{item['role']}】\n\n{item['content']}\n\n---\n"

    # 整理评分结果
    score = final_state.get("judge_score", {})
    if score:
        winner = "🏆 建议接受" if score.get("winner") == "affirmative" else "🏆 建议拒稿/大修"
        score_text = f"""## 📊 审稿评估结果

**{winner}**

| 评估维度 | 支持接受方 | 建议拒稿方 |
|------|------|------|
"""
        for dim in ["创新性评估", "方法论评估", "实验可靠性评估", "写作表达评估", "审稿全面性", "审稿深度", "审稿客观性", "立场坚定性"]:
            aff = score.get("affirmative_scores", {}).get(dim, "-")
            neg = score.get("negative_scores", {}).get(dim, "-")
            score_text += f"| {dim} | {aff} | {neg} |\n"

        score_text += f"| **总分** | **{score.get('affirmative_total', 0)}** | **{score.get('negative_total', 0)}** |\n\n"
        score_text += f"**分差**：{score.get('margin', 0)}\n\n"
        score_text += f"### 综合审稿意见\n\n{score.get('comment', '')}\n\n> 💡 以上为 AI 模拟审稿意见，仅供参考，最终审稿决定请以期刊/会议官方意见为准。\n"
    else:
        score_text = "审稿结果生成失败"

    # 完整文本
    full_text = format_debate_result(final_state)

    progress(1.0, desc="✅ 分析完成！")
    yield full_transcript, score_text, full_text, "✅ 分析完成！"


# ===== Gradio 界面 =====

with gr.Blocks(title="论文多视角审稿助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📝 论文多视角审稿助手

    基于 LangGraph 的多智能体论文审稿系统，输入论文标题/摘要/全文，AI 模拟审稿会讨论——支持接受审稿人与建议拒稿审稿人从创新性、方法论、实验、写作四个维度多轮攻防，最终输出综合审稿意见与修改建议。

    **审稿维度**：创新性 · 方法论 · 实验可靠性 · 写作表达

    > 💡 建议输入论文标题+摘要，或直接粘贴论文全文，审稿效果更好。
    """)

    # 示例论文快捷按钮
    with gr.Row():
        example_btn1 = gr.Button("🔬 大模型论文示例", size="sm")
        example_btn2 = gr.Button("📊 强化学习论文示例", size="sm")
        example_btn3 = gr.Button("🤖 机器人论文示例", size="sm")
        example_btn4 = gr.Button("📚 综述论文示例", size="sm")

    # ===== 输入区（无灰色底框，紧凑排列） =====
    topic_input = gr.Textbox(
        label="论文标题/摘要/全文",
        placeholder="粘贴论文标题、摘要或全文，例如：\n\n标题：基于多智能体强化学习的自动驾驶决策方法研究\n\n摘要：本文提出了一种...",
        lines=6,
        value="标题：基于多智能体强化学习的自动驾驶决策方法研究\n\n摘要：本文提出了一种基于多智能体强化学习的自动驾驶决策方法，通过引入注意力机制和课程学习策略，显著提升了复杂交通场景下的决策效率和安全性。实验结果表明，该方法在多个基准测试中优于现有方法。",
    )
    with gr.Row():
        affirmative_stance_input = gr.Textbox(
            label="支持接受方观点（选填）",
            placeholder="例如：论文创新性强，方法合理，实验充分",
            lines=1,
        )
        negative_stance_input = gr.Textbox(
            label="建议拒稿方观点（选填）",
            placeholder="例如：论文创新性不足，实验有缺陷，写作需改进",
            lines=1,
        )
    with gr.Row():
        rounds_slider = gr.Slider(
            minimum=1, maximum=5, value=2, step=1,
            label="审稿轮数",
        )
        reflection_checkbox = gr.Checkbox(
            value=True,
            label="自我反思（推荐）",
        )
        web_search_checkbox = gr.Checkbox(
            value=False,
            label="联网检索相关文献（Tavily）",
        )
        rag_checkbox = gr.Checkbox(
            value=False,
            label="离线参考文献检索（RAG）",
        )

    # RAG 文件上传（默认隐藏，勾选后显示）
    with gr.Row(visible=False) as rag_config:
        aff_evidence_input = gr.File(
            label="📄 支持接受方参考文献（可多选，支持 .txt/.md）",
            file_count="multiple",
            file_types=[".txt", ".md"],
        )
        neg_evidence_input = gr.File(
            label="📄 建议拒稿方参考文献（可多选，支持 .txt/.md）",
            file_count="multiple",
            file_types=[".txt", ".md"],
        )
    rag_checkbox.change(lambda x: gr.update(visible=x), inputs=rag_checkbox, outputs=rag_config)

    # 开始审稿按钮占满整行，状态显示在按钮下方
    run_button = gr.Button("🚀 开始审稿", variant="primary", size="lg")
    status_text = gr.Textbox(label="状态", value="等待开始...", interactive=False, lines=1)

    # ===== 输出区 =====
    # 用 State 保存完整 transcript 和展开状态
    full_transcript_state = gr.State("")
    is_expanded_state = gr.State(False)

    with gr.Row():
        with gr.Column(scale=3):
            transcript_md = gr.Markdown("审稿开始后这里会实时展示审稿讨论过程...", label="📜 审稿讨论过程")
            toggle_btn = gr.Button("展开全部", size="sm", visible=False)
        with gr.Column(scale=2):
            score_md = gr.Markdown("审稿结果会在审稿结束后展示...", label="📊 审稿评估")

    full_text_box = gr.Textbox(label="完整审稿记录（可全选复制）", lines=6, interactive=False)

    # 展开/收起切换函数
    def toggle_transcript(full_text, is_expanded):
        if is_expanded:
            # 当前是展开状态，点击后收起（只显示前面1000字）
            preview = full_text[:1000] if len(full_text) > 1000 else full_text
            if len(full_text) > 1000:
                preview = preview + "\n\n...（后面的内容已折叠，点击下方「展开全部」查看完整辩论记录）"
            return preview, False, "展开全部"
        else:
            # 当前是收起状态，点击后展开
            return full_text, True, "收起"

    # 事件绑定
    run_button.click(
        fn=run_debate_ui,
        inputs=[topic_input, rounds_slider, reflection_checkbox,
                affirmative_stance_input, negative_stance_input,
                rag_checkbox, aff_evidence_input, neg_evidence_input,
                web_search_checkbox],
        outputs=[transcript_md, score_md, full_text_box, status_text],
    )

    # 辩论完成后自动折叠 transcript（只显示前面部分），显示展开按钮
    def on_status_change(status_text, full_transcript):
        if status_text and "分析完成" in status_text and full_transcript and len(full_transcript) > 1000:
            # 分析完成，折叠成只显示前面1000字
            preview = full_transcript[:1000] + "\n\n...（后面的内容已折叠，点击下方「展开全部」查看完整分析记录）"
            return preview, full_transcript, False, gr.update(visible=True, value="展开全部")
        return full_transcript, full_transcript, False, gr.update(visible=False)

    status_text.change(
        fn=on_status_change,
        inputs=[status_text, transcript_md],
        outputs=[transcript_md, full_transcript_state, is_expanded_state, toggle_btn],
    )

    # 展开/收起按钮
    toggle_btn.click(
        fn=toggle_transcript,
        inputs=[full_transcript_state, is_expanded_state],
        outputs=[transcript_md, is_expanded_state, toggle_btn],
    )

    # 示例论文快捷按钮
    def fill_example(text):
        return text
    example_btn1.click(fn=fill_example, inputs=[gr.State("标题：基于大语言模型的代码自动生成方法研究\n\n摘要：本文提出了一种基于大语言模型的代码自动生成方法，通过引入检索增强生成和程序语义约束，显著提升了生成代码的正确率和可执行性。实验结果表明，该方法在多个代码生成基准测试中优于现有方法。")], outputs=[topic_input])
    example_btn2.click(fn=fill_example, inputs=[gr.State("标题：基于深度强化学习的机器人运动控制方法研究\n\n摘要：本文提出了一种基于深度强化学习的机器人运动控制方法，通过引入课程学习和奖励塑形策略，解决了传统方法在复杂环境下样本效率低的问题。仿真实验表明，该方法能够有效提升机器人的运动性能。")], outputs=[topic_input])
    example_btn3.click(fn=fill_example, inputs=[gr.State("标题：基于多智能体协同的自动驾驶决策系统研究\n\n摘要：本文设计了一种基于多智能体协同的自动驾驶决策系统，通过车路协同和意图预测机制，提升了复杂交通场景下的决策安全性和通行效率。实车测试结果表明，该系统在多种场景下均表现良好。")], outputs=[topic_input])
    example_btn4.click(fn=fill_example, inputs=[gr.State("标题：大语言模型在自然语言处理中的应用综述\n\n摘要：本文综述了大语言模型在自然语言处理领域的最新进展，包括预训练方法、指令微调、对齐技术等方面，并讨论了当前面临的挑战和未来研究方向。")], outputs=[topic_input])

    # 页脚
    gr.Markdown("""
    ---
    <sub>**技术栈**：LangGraph · LangChain · OpenAI 兼容 API（智谱 GLM / DeepSeek / GPT）· FAISS · Tavily · Gradio</sub>
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
