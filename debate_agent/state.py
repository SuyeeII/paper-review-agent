"""辩论系统状态定义"""
from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum


class DebatePhase(str, Enum):
    """辩论阶段"""
    OPENING = "opening"           # 立论
    CLASH = "clash"               # 攻辩（多轮）
    REBUTTAL = "rebuttal"         # 驳论
    CLOSING = "closing"           # 总结
    JUDGING = "judging"           # 评委评分
    DONE = "done"                  # 结束


class DebateState(TypedDict):
    """
    辩论全局状态
    LangGraph 中所有节点共享这个 state dict
    """
    # ===== 基本信息 =====
    topic: str                          # 辩题
    affirmative_stance: str             # 正方立场的明确表述（如"AI会取代程序员"）
    negative_stance: str                # 反方立场的明确表述（如"AI不会取代程序员"）
    max_rounds: int                     # 最大攻辩轮数
    current_round: int                  # 当前攻辩轮次
    phase: DebatePhase                  # 当前阶段
    current_speaker: str                # 当前发言方 "affirmative" / "negative" / "judge"

    # ===== 辩论历史 =====
    affirmative_opening: Optional[str]  # 正方立论
    negative_opening: Optional[str]     # 反方立论
    affirmative_clashes: List[str]      # 正方各轮攻辩发言
    negative_clashes: List[str]         # 反方各轮攻辩发言
    affirmative_rebuttal: Optional[str] # 正方驳论
    negative_rebuttal: Optional[str]    # 反方驳论
    affirmative_closing: Optional[str]   # 正方总结
    negative_closing: Optional[str]      # 反方总结

    # ===== Reflection 机制 =====
    affirmative_reflection: Optional[str]  # 正方自我反思
    negative_reflection: Optional[str]     # 反方自我反思
    reflection_enabled: bool                # 是否启用自我反思

    # ===== V1.5 记忆摘要压缩 =====
    affirmative_summary: Optional[str]     # 正方历史发言摘要（久远轮次压缩后）
    negative_summary: Optional[str]        # 反方历史发言摘要
    summary_enabled: bool                   # 是否启用记忆摘要压缩
    summary_threshold: int                  # 超过多少轮后开始摘要（默认2，即第3轮开始对第1轮做摘要）

    # ===== 评委评分 =====
    judge_score: Optional[Dict[str, Any]]  # 评委评分结果
    winner: Optional[str]                    # 获胜方

    # ===== 系统 =====
    error: Optional[str]                     # 错误信息
    full_transcript: List[Dict[str, str]]   # 完整对话记录 [{speaker, role, content}]


def init_state(
    topic: str,
    max_rounds: int = 3,
    reflection_enabled: bool = True,
    affirmative_stance: str = None,
    negative_stance: str = None,
    summary_enabled: bool = True,
    summary_threshold: int = 2,
) -> DebateState:
    """
    初始化辩论状态

    Args:
        topic: 辩题
        max_rounds: 最大攻辩轮数
        reflection_enabled: 是否启用自我反思
        affirmative_stance: 正方立场的明确表述（如"猫更适合当宠物"），不填则默认"支持本辩题"
        negative_stance: 反方立场的明确表述（如"狗更适合当宠物"），不填则默认"反对本辩题"
        summary_enabled: 是否启用记忆摘要压缩（V1.5）
        summary_threshold: 超过多少轮后开始摘要（默认2，即第3轮开始对第1轮做摘要）
    """
    return {
        "topic": topic,
        "affirmative_stance": affirmative_stance or "支持本辩题的立场",
        "negative_stance": negative_stance or "反对本辩题的立场",
        "max_rounds": max_rounds,
        "current_round": 0,
        "phase": DebatePhase.OPENING,
        "current_speaker": "affirmative",
        "affirmative_opening": None,
        "negative_opening": None,
        "affirmative_clashes": [],
        "negative_clashes": [],
        "affirmative_rebuttal": None,
        "negative_rebuttal": None,
        "affirmative_closing": None,
        "negative_closing": None,
        "affirmative_reflection": None,
        "negative_reflection": None,
        "reflection_enabled": reflection_enabled,
        "affirmative_summary": None,
        "negative_summary": None,
        "summary_enabled": summary_enabled,
        "summary_threshold": summary_threshold,
        "judge_score": None,
        "winner": None,
        "error": None,
        "full_transcript": [],
    }
