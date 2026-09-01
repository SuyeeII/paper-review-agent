"""论文多视角审稿助手图定义（多Agent并行评审架构）
4个维度审稿人并行评审 → 自我反思修正 → 主编汇总
"""
from typing import Optional, Callable, List
from langgraph.graph import StateGraph, END
from debate_agent.state import ReviewState, ReviewPhase, init_state
from debate_agent import nodes


def node_review_done(state: ReviewState) -> ReviewState:
    """汇聚节点：等待4个审稿节点都完成后，什么都不做，直接返回"""
    return state


def node_start_reflection(state: ReviewState) -> ReviewState:
    """开始反思修正阶段的节点：什么都不做，只是出发到4个并行反思修正节点"""
    return state


def node_reflection_done(state: ReviewState) -> ReviewState:
    """汇聚节点：等待4个反思修正节点都完成后，什么都不做，直接返回"""
    return state


def build_graph() -> StateGraph:
    """
    构建审稿流程图

    流程：
    START → 论文结构解析 → 4个并行审稿节点 → 汇聚节点 → 路由判断
        ├─ reflection_enabled → 开始反思节点 → 4个并行反思修正节点 → 汇聚节点 → 主编汇总 → END
        └─ !reflection_enabled → 主编汇总 → END
    """
    workflow = StateGraph(ReviewState)

    # ===== 节点定义 =====
    # 论文结构解析节点（第一步）
    workflow.add_node("paper_structure", nodes.node_paper_structure)

    # 4个维度审稿节点（并行）
    workflow.add_node("innovation_review", nodes.node_innovation_review)
    workflow.add_node("methodology_review", nodes.node_methodology_review)
    workflow.add_node("experiment_review", nodes.node_experiment_review)
    workflow.add_node("writing_review", nodes.node_writing_review)

    # 汇聚节点：等待4个审稿节点都完成
    workflow.add_node("review_done", node_review_done)

    # 4个维度反思修正节点（并行，仅当reflection_enabled时执行）
    workflow.add_node("innovation_reflection", nodes.node_innovation_reflection)
    workflow.add_node("methodology_reflection", nodes.node_methodology_reflection)
    workflow.add_node("experiment_reflection", nodes.node_experiment_reflection)
    workflow.add_node("writing_reflection", nodes.node_writing_reflection)

    # 反思阶段开始节点和汇聚节点
    workflow.add_node("start_reflection", node_start_reflection)
    workflow.add_node("reflection_done", node_reflection_done)

    # 主编汇总节点
    workflow.add_node("editor_summary", nodes.node_editor_summary)

    # ===== 边定义 =====
    # 起点 → 论文结构解析
    workflow.set_entry_point("paper_structure")

    # 论文结构解析 → 4个并行审稿节点
    workflow.add_edge("paper_structure", "innovation_review")
    workflow.add_edge("paper_structure", "methodology_review")
    workflow.add_edge("paper_structure", "experiment_review")
    workflow.add_edge("paper_structure", "writing_review")

    # 4个审稿节点都完成后 → 汇聚节点
    workflow.add_edge("innovation_review", "review_done")
    workflow.add_edge("methodology_review", "review_done")
    workflow.add_edge("experiment_review", "review_done")
    workflow.add_edge("writing_review", "review_done")

    # 汇聚节点 → 路由判断（是否需要反思修正）
    workflow.add_conditional_edges(
        "review_done",
        nodes.route_after_review,
        {
            "reflection": "start_reflection",
            "summary": "editor_summary",
        },
    )

    # 开始反思节点 → 4个并行反思修正节点
    workflow.add_edge("start_reflection", "innovation_reflection")
    workflow.add_edge("start_reflection", "methodology_reflection")
    workflow.add_edge("start_reflection", "experiment_reflection")
    workflow.add_edge("start_reflection", "writing_reflection")

    # 4个反思修正节点都完成后 → 汇聚节点
    workflow.add_edge("innovation_reflection", "reflection_done")
    workflow.add_edge("methodology_reflection", "reflection_done")
    workflow.add_edge("experiment_reflection", "reflection_done")
    workflow.add_edge("writing_reflection", "reflection_done")

    # 反思汇聚节点 → 主编汇总
    workflow.add_edge("reflection_done", "editor_summary")

    # 主编汇总 → 结束
    workflow.add_edge("editor_summary", END)

    return workflow


# 全局图实例
_graph = None


def get_graph() -> StateGraph:
    """获取全局图实例（懒加载）"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_review(
    topic: str,
    reflection_enabled: bool = True,
    rag_enabled: bool = False,
    web_search_enabled: bool = False,
    progress_callback: Optional[Callable[[str, ReviewState], None]] = None,
) -> ReviewState:
    """
    运行一次完整的论文审稿

    Args:
        topic: 论文内容
        reflection_enabled: 是否启用自我反思机制
        rag_enabled: 是否启用 RAG 参考文献检索（后续实现）
        web_search_enabled: 是否启用 Tavily 联网检索（后续实现）
        progress_callback: 进度回调函数

    Returns:
        最终的 ReviewState，包含所有审稿意见和主编汇总
    """
    # 初始化状态
    state = init_state(
        topic=topic,
        reflection_enabled=reflection_enabled,
        rag_enabled=rag_enabled,
        web_search_enabled=web_search_enabled,
    )

    # 构建并运行图
    graph = get_graph()
    app = graph.compile()
    final_state = app.invoke(state)

    return final_state


def format_review_result(state: ReviewState) -> str:
    """
    格式化审稿结果为可读文本

    Args:
        state: 审稿完成后的状态

    Returns:
        格式化后的完整审稿报告文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("📝 论文多视角审稿报告")
    lines.append("=" * 60)
    lines.append("")

    # 论文结构解析结果
    if state.get("paper_structure"):
        lines.append("【论文结构解析】")
        lines.append(state["paper_structure"])
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    # 论文内容（截断显示）
    paper_preview = state["topic"][:500] + "..." if len(state["topic"]) > 500 else state["topic"]
    lines.append(f"【论文内容】\n{paper_preview}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("")

    # 4个维度审稿意见
    dimensions = [
        ("创新性", state.get("innovation_final") or state.get("innovation_review")),
        ("方法论", state.get("methodology_final") or state.get("methodology_review")),
        ("实验可靠性", state.get("experiment_final") or state.get("experiment_review")),
        ("写作表达", state.get("writing_final") or state.get("writing_review")),
    ]

    for dim_name, review in dimensions:
        if review:
            lines.append(f"【{dim_name}审稿人】")
            lines.append(review)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

    # 主编综合审稿报告
    if state.get("editor_summary"):
        lines.append("【主编综合审稿报告】")
        lines.append(state["editor_summary"])
        lines.append("")

    lines.append("=" * 60)
    lines.append("💡 以上为 AI 模拟审稿意见，仅供参考，最终审稿决定请以期刊/会议官方意见为准。")

    return "\n".join(lines)
