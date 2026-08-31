"""LangGraph 辩论图构建与运行入口
这是整个系统的核心编排层
"""
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END

from .state import DebateState, init_state
from . import nodes


def create_debate_graph() -> StateGraph:
    """
    构建辩论流程图

    流程：
    正方立论 → 反方立论 → [攻辩循环] → 正方驳论 → 反方驳论 → 正方总结 → 反方总结 → 评委评分 → END

    攻辩循环：
    正方攻辩 → 反方攻辩 → 自我反思 → 记忆摘要压缩(V1.5) → (判断轮数) → 继续攻辩 / 进入驳论
    """
    graph = StateGraph(DebateState)

    # ===== 添加所有节点 =====
    graph.add_node("opening_affirmative", nodes.node_opening_affirmative)
    graph.add_node("opening_negative", nodes.node_opening_negative)
    graph.add_node("clash_affirmative", nodes.node_clash_affirmative)
    graph.add_node("clash_negative", nodes.node_clash_negative)
    graph.add_node("reflection", nodes.node_reflection)
    graph.add_node("summary", nodes.node_summary)  # V1.5 记忆摘要压缩
    graph.add_node("rebuttal_affirmative", nodes.node_rebuttal_affirmative)
    graph.add_node("rebuttal_negative", nodes.node_rebuttal_negative)
    graph.add_node("closing_affirmative", nodes.node_closing_affirmative)
    graph.add_node("closing_negative", nodes.node_closing_negative)
    graph.add_node("judge", nodes.node_judge)

    # ===== 设置入口 =====
    graph.set_entry_point("opening_affirmative")

    # ===== 线性边 =====
    graph.add_edge("opening_affirmative", "opening_negative")
    graph.add_edge("opening_negative", "clash_affirmative")
    graph.add_edge("clash_affirmative", "clash_negative")
    graph.add_edge("clash_negative", "reflection")
    graph.add_edge("reflection", "summary")  # V1.5 反思后做摘要压缩

    # ===== 条件边：攻辩循环的核心 =====
    # 摘要结束后，判断是否继续攻辩
    graph.add_conditional_edges(
        "summary",
        nodes.should_continue_clash,
        {
            # 继续下一轮攻辩
            "clash_affirmative": "clash_affirmative",
            # 攻辩结束，进入驳论
            "rebuttal_affirmative": "rebuttal_affirmative",
        },
    )

    # ===== 驳论 → 总结 → 评分 =====
    graph.add_edge("rebuttal_affirmative", "rebuttal_negative")
    graph.add_edge("rebuttal_negative", "closing_affirmative")
    graph.add_edge("closing_affirmative", "closing_negative")
    graph.add_edge("closing_negative", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_debate(
    topic: str,
    max_rounds: int = 3,
    reflection_enabled: bool = True,
    affirmative_stance: str = None,
    negative_stance: str = None,
    progress_callback: Optional[Callable[[str, DebateState], None]] = None,
) -> DebateState:
    """
    运行一场完整辩论

    Args:
        topic: 辩题
        max_rounds: 最大攻辩轮数（默认3轮）
        reflection_enabled: 是否启用自我反思机制（核心算法亮点）
        affirmative_stance: 正方立场的明确表述（如"猫更适合当宠物"），不填则默认"支持本辩题"
        negative_stance: 反方立场的明确表述（如"狗更适合当宠物"），不填则默认"反对本辩题"
        progress_callback: 进度回调函数，签名 callback(current_stage, state)，用于前端实时展示

    Returns:
        最终的 DebateState，包含所有发言和评委评分
    """
    # 初始化状态
    state = init_state(topic, max_rounds, reflection_enabled, affirmative_stance, negative_stance)

    # 编译图
    graph = create_debate_graph()

    # 如果有进度回调，用 stream 模式逐步执行
    if progress_callback:
        final_state = None
        for output in graph.stream(state):
            for node_name, node_state in output.items():
                progress_callback(node_name, node_state)
                final_state = node_state
        return final_state
    else:
        # 无回调，直接执行到底
        return graph.invoke(state)


def format_debate_result(state: DebateState) -> str:
    """
    将辩论结果格式化为可读文本
    用于终端输出或保存到文件
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"辩题：{state['topic']}")
    lines.append(f"攻辩轮数：{state['max_rounds']}")
    lines.append(f"自我反思：{'启用' if state['reflection_enabled'] else '关闭'}")
    lines.append("=" * 60)
    lines.append("")

    for item in state["full_transcript"]:
        lines.append(f"【{item['role']}】")
        lines.append(item["content"])
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    # 评分结果
    if state["judge_score"]:
        score = state["judge_score"]
        lines.append("=" * 60)
        lines.append("📊 评委评分结果")
        lines.append("=" * 60)
        lines.append(f"正方总分：{score.get('affirmative_total', 0)} / 80")
        lines.append(f"反方总分：{score.get('negative_total', 0)} / 80")
        winner = "正方" if score.get("winner") == "affirmative" else "反方"
        lines.append(f"🏆 获胜方：{winner}")
        lines.append(f"分差：{score.get('margin', 0)}")
        lines.append("")
        lines.append("各维度得分：")
        for dim in ["立论深度", "逻辑论证", "论据质量", "反驳能力", "应变能力", "语言表达", "整体配合", "立场坚定性"]:
            aff = score.get("affirmative_scores", {}).get(dim, "-")
            neg = score.get("negative_scores", {}).get(dim, "-")
            lines.append(f"  {dim}：正方 {aff} | 反方 {neg}")
        lines.append("")
        lines.append("评委点评：")
        lines.append(score.get("comment", ""))

    return "\n".join(lines)
