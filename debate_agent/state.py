"""论文多视角审稿助手状态定义（多Agent并行评审架构）"""
from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum


class ReviewPhase(str, Enum):
    """审稿阶段"""
    REVIEWING = "reviewing"        # 4个审稿人并行评审
    REFLECTING = "reflecting"      # 自我反思修正
    SUMMARIZING = "summarizing"    # 主编汇总
    DONE = "done"                   # 结束


class ReviewState(TypedDict):
    """
    审稿全局状态
    LangGraph 中所有节点共享这个 state dict
    """
    # ===== 基本信息 =====
    topic: str                          # 论文内容
    reflection_enabled: bool            # 是否启用自我反思机制

    # ===== 4个维度审稿意见（初审）=====
    innovation_review: Optional[str]    # 创新性审稿意见
    methodology_review: Optional[str]   # 方法论审稿意见
    experiment_review: Optional[str]    # 实验审稿意见（含可复现性）
    writing_review: Optional[str]       # 写作审稿意见

    # ===== 4个维度审稿意见（反思修正后，最终版）=====
    innovation_final: Optional[str]     # 创新性最终审稿意见
    methodology_final: Optional[str]    # 方法论最终审稿意见
    experiment_final: Optional[str]     # 实验最终审稿意见
    writing_final: Optional[str]        # 写作最终审稿意见

    # ===== 主编汇总 =====
    editor_summary: Optional[str]        # 主编综合审稿报告

    # ===== 增强功能（后续实现）=====
    rag_enabled: bool                    # 是否启用 RAG 参考文献检索
    web_search_enabled: bool             # 是否启用 Tavily 联网检索
    paper_structure: Optional[Dict[str, str]]  # 论文结构解析结果（后续实现）

    # ===== 系统 =====
    phase: ReviewPhase                   # 当前阶段
    error: Optional[str]                 # 错误信息
    full_transcript: List[Dict[str, str]]  # 完整记录 [{reviewer, role, content}]


def init_state(
    topic: str,
    reflection_enabled: bool = True,
    rag_enabled: bool = False,
    web_search_enabled: bool = False,
) -> ReviewState:
    """
    初始化审稿状态

    Args:
        topic: 论文内容
        reflection_enabled: 是否启用自我反思机制
        rag_enabled: 是否启用 RAG 参考文献检索
        web_search_enabled: 是否启用 Tavily 联网检索
    """
    return {
        "topic": topic,
        "reflection_enabled": reflection_enabled,
        "innovation_review": None,
        "methodology_review": None,
        "experiment_review": None,
        "writing_review": None,
        "innovation_final": None,
        "methodology_final": None,
        "experiment_final": None,
        "writing_final": None,
        "editor_summary": None,
        "rag_enabled": rag_enabled,
        "web_search_enabled": web_search_enabled,
        "paper_structure": None,
        "phase": ReviewPhase.REVIEWING,
        "error": None,
        "full_transcript": [],
    }
