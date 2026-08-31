"""多 Agent 辩论系统 - Gradio 界面
运行：python app.py
然后浏览器打开 http://localhost:7860
"""
import gradio as gr
from debate_agent import run_debate, format_debate_result
from debate_agent.state import DebateState


def run_debate_ui(topic: str, max_rounds: int, reflection_enabled: bool, affirmative_stance: str, negative_stance: str, progress=gr.Progress()):
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
            "opening_affirmative": "正方立论中...",
            "opening_negative": "反方立论中...",
            "clash_affirmative": f"正方第{state.get('current_round', 1)}轮攻辩中...",
            "clash_negative": f"反方第{state.get('current_round', 1)}轮攻辩中...",
            "reflection": "双方自我反思中...",
            "summary": "记忆摘要压缩中...",
            "rebuttal_affirmative": "正方驳论中...",
            "rebuttal_negative": "反方驳论中...",
            "closing_affirmative": "正方总结陈词中...",
            "closing_negative": "反方总结陈词中...",
            "judge": "评委评分中...",
        }
        progress(0.5, desc=stage_names.get(node_name, "辩论进行中..."))

    # 运行辩论
    try:
        progress(0.1, desc="初始化辩论系统...")
        final_state = run_debate(
            topic=topic.strip(),
            max_rounds=int(max_rounds),
            reflection_enabled=reflection_enabled,
            affirmative_stance=aff_stance,
            negative_stance=neg_stance,
            progress_callback=progress_callback,
        )
    except Exception as e:
        error_msg = f"❌ 辩论运行出错：{str(e)}"
        yield error_msg, "", "", ""
        return

    # 整理完整 transcript
    full_transcript = ""
    for item in final_state["full_transcript"]:
        full_transcript += f"### 【{item['role']}】\n\n{item['content']}\n\n---\n"

    # 整理评分结果
    score = final_state.get("judge_score", {})
    if score:
        winner = "🏆 正方获胜" if score.get("winner") == "affirmative" else "🏆 反方获胜"
        score_text = f"""## 📊 评委评分结果

**{winner}**

| 维度 | 正方 | 反方 |
|------|------|------|
"""
        for dim in ["立论深度", "逻辑论证", "论据质量", "反驳能力", "应变能力", "语言表达", "整体配合", "立场坚定性"]:
            aff = score.get("affirmative_scores", {}).get(dim, "-")
            neg = score.get("negative_scores", {}).get(dim, "-")
            score_text += f"| {dim} | {aff} | {neg} |\n"

        score_text += f"| **总分** | **{score.get('affirmative_total', 0)}** | **{score.get('negative_total', 0)}** |\n\n"
        score_text += f"**分差**：{score.get('margin', 0)}\n\n"
        score_text += f"### 评委点评\n\n{score.get('comment', '')}\n"
    else:
        score_text = "评分结果生成失败"

    # 完整文本
    full_text = format_debate_result(final_state)

    progress(1.0, desc="✅ 辩论完成！")
    yield full_transcript, score_text, full_text, "✅ 辩论完成！"


# ===== Gradio 界面 =====

with gr.Blocks(title="多 Agent 辩论系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 多 Agent 辩论系统

    基于 LangGraph 的多智能体辩论系统，支持**立论 → 攻辩（多轮）→ 自我反思 → 驳论 → 总结 → 评委评分**完整流程。

    核心算法亮点：**Self-Reflection 自我反思机制** —— 每轮攻辩后，双方 Agent 会批判自己之前的论证，找出漏洞并在下一轮修正。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="辩题",
                placeholder="例如：AI 会不会取代程序员？\n猫和狗谁更适合当宠物？\n年轻人应不应该躺平？",
                lines=3,
                value="AI 会不会取代程序员？",
            )
            with gr.Row():
                affirmative_stance_input = gr.Textbox(
                    label="正方立场（选填，强烈建议填写，防止立场跑偏）",
                    placeholder="例如：AI会取代程序员 / 猫更适合当宠物",
                    lines=1,
                )
                negative_stance_input = gr.Textbox(
                    label="反方立场（选填，强烈建议填写，防止立场跑偏）",
                    placeholder="例如：AI不会取代程序员 / 狗更适合当宠物",
                    lines=1,
                )
            rounds_slider = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="攻辩轮数（轮数越多辩论越精彩，但耗时越长）",
            )
            reflection_checkbox = gr.Checkbox(
                value=True,
                label="启用自我反思机制（Self-Reflection）✅ 推荐开启",
            )
            run_button = gr.Button("🚀 开始辩论", variant="primary", size="lg")

            status_text = gr.Textbox(label="状态", value="等待开始...", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📜 辩论实录"):
                    transcript_md = gr.Markdown("辩论开始后这里会实时展示双方发言...")
                with gr.Tab("📊 评分结果"):
                    score_md = gr.Markdown("评分结果会在辩论结束后展示...")

    with gr.Row():
        full_text_box = gr.Textbox(label="完整辩论记录（可全选复制）", lines=8, interactive=False)

    # 事件绑定
    run_button.click(
        fn=run_debate_ui,
        inputs=[topic_input, rounds_slider, reflection_checkbox, affirmative_stance_input, negative_stance_input],
        outputs=[transcript_md, score_md, full_text_box, status_text],
    )

    gr.Markdown("""
    ---
    **技术栈**：LangGraph（多 Agent 编排）| LangChain | OpenAI 兼容 API（DeepSeek / Qwen / GPT）| Gradio

    **项目结构**：
    - `debate_agent/state.py` — 辩论状态定义
    - `debate_agent/prompts.py` — Prompt 模板（立论/攻辩/反思/驳论/总结/评分）
    - `debate_agent/debater.py` — 辩手 Agent
    - `debate_agent/judge.py` — 评委 Agent
    - `debate_agent/nodes.py` — LangGraph 图节点
    - `debate_agent/graph.py` — 辩论图构建与运行
    - `app.py` — Gradio 界面
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
