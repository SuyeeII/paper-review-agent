"""论文多视角审稿助手状态定义（多Agent并行评审架构）"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from enum import Enum
import operator


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

    # ===== 4个维度审稿意见 =====
    innovation_review: Optional[str]    # 创新性审稿意见
    methodology_review: Optional[str]   # 方法论审稿意见
    experiment_review: Optional[str]    # 论证与证据审稿意见
    writing_review: Optional[str]       # 写作审稿意见

    # ===== 主编汇总 =====
    editor_summary: Optional[str]        # 主编综合审稿报告

    # ===== 论文结构解析 =====
    paper_structure: Optional[str]       # 论文结构解析结果

    # ===== PDF图表解析信息 =====
    tables_md: Optional[str]             # 表格 Markdown 文本（pdfplumber 提取）
    images: int                          # 图片数量
    tables: int                          # 表格数量

    # ===== 系统 =====
    phase: ReviewPhase                   # 当前阶段
    error: Optional[str]                 # 错误信息
    full_transcript: Annotated[List[Dict[str, str]], operator.add]  # 完整记录，并行节点追加时用operator.add合并


def init_state(
    topic: str,
    tables_md: str = "",
    images: int = 0,
    tables: int = 0,
) -> ReviewState:
    """
    初始化审稿状态

    Args:
        topic: 论文内容
        tables_md: 表格 Markdown 文本（pdfplumber 提取，可为空）
        images: PDF 中图片数量
        tables: PDF 中表格数量
    """
    return {
        "topic": topic,
        "innovation_review": None,
        "methodology_review": None,
        "experiment_review": None,
        "writing_review": None,
        "editor_summary": None,
        "paper_structure": None,
        "tables_md": tables_md,
        "images": images,
        "tables": tables,
        "phase": ReviewPhase.REVIEWING,
        "error": None,
        "full_transcript": [],
    }
