"""LangGraph 图节点函数
每个节点接收 state，处理后返回更新后的 state
V2 新增：RAG 论据检索集成
"""
from typing import Dict, Any, Optional
from .state import DebateState, DebatePhase
from .debater import DebaterAgent
from .judge import JudgeAgent
from .rag import KnowledgeBase
from .web_search import TavilySearch

# V2 RAG 全局知识库（由 run_debate 设置，节点里读取）
_affirmative_kb: Optional[KnowledgeBase] = None
_negative_kb: Optional[KnowledgeBase] = None

# V3 联网检索全局实例（由 run_debate 设置，节点里读取）
_web_search: Optional[TavilySearch] = None


def set_evidence_knowledge_bases(aff_kb: Optional[KnowledgeBase], neg_kb: Optional[KnowledgeBase]):
    """设置正方/反方的论据知识库（V2 RAG）"""
    global _affirmative_kb, _negative_kb
    _affirmative_kb = aff_kb
    _negative_kb = neg_kb


def set_web_search(web_search: Optional[TavilySearch]):
    """设置联网检索实例（V3 Tavily）"""
    global _web_search
    _web_search = web_search


def _get_debater(side: str, stance_detail: str) -> DebaterAgent:
    """创建辩手 Agent 并设置论据知识库和联网检索（V2 RAG + V3 联网检索）"""
    debater = DebaterAgent(side, stance_detail)
    if side == "affirmative" and _affirmative_kb is not None:
        debater.evidence_kb = _affirmative_kb
    elif side == "negative" and _negative_kb is not None:
        debater.evidence_kb = _negative_kb
    if _web_search is not None:
        debater.web_search = _web_search
    return debater


# ===== 立论阶段 =====

def node_opening_affirmative(state: DebateState) -> DebateState:
    """正方立论"""
    debater = _get_debater("affirmative", state["affirmative_stance"])
    speech = debater.opening(state["topic"])
    state["affirmative_opening"] = speech
    state["current_speaker"] = "negative"
    state["full_transcript"].append({
        "speaker": "affirmative",
        "role": "正方一辩（立论）",
        "content": speech,
    })
    return state


def node_opening_negative(state: DebateState) -> DebateState:
    """反方立论"""
    debater = _get_debater("negative", state["negative_stance"])
    speech = debater.opening(state["topic"])
    state["negative_opening"] = speech
    state["phase"] = DebatePhase.CLASH
    state["current_speaker"] = "affirmative"
    state["current_round"] = 1
    state["full_transcript"].append({
        "speaker": "negative",
        "role": "反方一辩（立论）",
        "content": speech,
    })
    return state


# ===== 攻辩阶段（多轮循环核心）=====

def node_clash_affirmative(state: DebateState) -> DebateState:
    """正方攻辩"""
    # V1.5 修复：轮数增加从条件边函数移到这里
    # 条件边函数(router)对state的修改不会被保留，所以必须在节点里做
    # 如果不是第1轮攻辩（已有正方攻辩记录），轮数+1
    if state["affirmative_clashes"]:
        state["current_round"] += 1

    round_num = state["current_round"]

    # 确定"对方上一轮发言"和"自己上一轮发言"
    if round_num == 1:
        # 第一轮攻辩，对方上一轮是立论，自己上一轮也是立论
        opponent_prev = state["negative_opening"]
        own_prev = state["affirmative_opening"]
    else:
        opponent_prev = state["negative_clashes"][-1]
        own_prev = state["affirmative_clashes"][-1]

    # 如果启用了反思且上一轮有反思结果，传入
    reflection = state.get("affirmative_reflection") if state.get("reflection_enabled") else None

    debater = _get_debater("affirmative", state["affirmative_stance"])
    speech = debater.clash(
        topic=state["topic"],
        round_num=round_num,
        opponent_previous=opponent_prev,
        own_previous=own_prev,
        reflection=reflection,
    )

    state["affirmative_clashes"].append(speech)
    state["current_speaker"] = "negative"
    # 用完反思后清空，下一轮会重新生成
    state["affirmative_reflection"] = None
    state["full_transcript"].append({
        "speaker": "affirmative",
        "role": f"正方（第{round_num}轮攻辩）",
        "content": speech,
    })
    return state


def node_clash_negative(state: DebateState) -> DebateState:
    """反方攻辩"""
    round_num = state["current_round"]

    # 反方的"对方上一轮"就是刚发生的正方攻辩
    opponent_prev = state["affirmative_clashes"][-1]

    if round_num == 1:
        own_prev = state["negative_opening"]
    else:
        own_prev = state["negative_clashes"][-1]

    reflection = state.get("negative_reflection") if state.get("reflection_enabled") else None

    debater = _get_debater("negative", state["negative_stance"])
    speech = debater.clash(
        topic=state["topic"],
        round_num=round_num,
        opponent_previous=opponent_prev,
        own_previous=own_prev,
        reflection=reflection,
    )

    state["negative_clashes"].append(speech)
    state["current_speaker"] = "affirmative"
    state["negative_reflection"] = None
    state["full_transcript"].append({
        "speaker": "negative",
        "role": f"反方（第{round_num}轮攻辩）",
        "content": speech,
    })
    return state


# ===== Self-Reflection 节点（核心算法亮点）=====

def node_reflection(state: DebateState) -> DebateState:
    """
    双方自我反思
    每轮攻辩结束后调用，双方各自批判自己的论证
    反思结果会在下一轮攻辩中使用
    """
    if not state.get("reflection_enabled"):
        return state

    # 收集双方到目前为止的所有发言（V1.5：如果有摘要，久远部分用摘要代替）
    aff_speeches = _get_side_speeches_for_context(state, "affirmative")
    neg_speeches = _get_side_speeches_for_context(state, "negative")

    # 正方反思（参考自己的所有发言 + 反方的所有发言）
    aff_debater = _get_debater("affirmative", state["affirmative_stance"])
    aff_reflection = aff_debater.reflect(state["topic"], aff_speeches, neg_speeches)
    state["affirmative_reflection"] = aff_reflection

    # 反方反思（参考自己的所有发言 + 正方的所有发言）
    neg_debater = _get_debater("negative", state["negative_stance"])
    neg_reflection = neg_debater.reflect(state["topic"], neg_speeches, aff_speeches)
    state["negative_reflection"] = neg_reflection

    state["full_transcript"].append({
        "speaker": "system",
        "role": "【自我反思】正方",
        "content": aff_reflection,
    })
    state["full_transcript"].append({
        "speaker": "system",
        "role": "【自我反思】反方",
        "content": neg_reflection,
    })

    return state


# ===== 攻辩轮数判断（条件边）=====

def should_continue_clash(state: DebateState) -> str:
    """
    条件边函数：判断是否继续攻辩
    返回下一个节点的名称
    注意：条件边函数(router)对state的修改不会被保留，所以这里只做判断，不修改state
    轮数增加已移到 node_clash_affirmative 节点开头
    """
    if state["current_round"] >= state["max_rounds"]:
        # 攻辩结束，进入驳论
        return "rebuttal_affirmative"
    else:
        # 继续下一轮攻辩（轮数+1在 node_clash_affirmative 开头处理）
        return "clash_affirmative"


# ===== V1.5 记忆摘要压缩节点 =====

def _get_side_speeches_for_context(state: DebateState, side: str, include_rebuttal: bool = False) -> list:
    """
    V1.5 辅助函数：获取一方的发言列表，用于上下文拼接
    如果启用了摘要且有摘要，久远部分用摘要代替，减少 token 消耗

    Args:
        state: 辩论状态
        side: "affirmative" 正方 / "negative" 反方
        include_rebuttal: 是否包含驳论发言（总结阶段需要）
    """
    prefix = side
    threshold = state.get("summary_threshold", 2)

    # 基础发言：立论 + 所有攻辩
    speeches = []
    if state.get(f"{prefix}_opening"):
        speeches.append(state[f"{prefix}_opening"])
    speeches.extend(state.get(f"{prefix}_clashes", []))

    # 驳论（总结阶段需要）
    if include_rebuttal and state.get(f"{prefix}_rebuttal"):
        speeches.append(state[f"{prefix}_rebuttal"])

    # 如果启用了摘要且有摘要，且发言数超过阈值，久远部分用摘要代替
    if state.get("summary_enabled") and state.get(f"{prefix}_summary") and len(speeches) > threshold:
        recent_speeches = speeches[-threshold:]
        return [f"【历史发言摘要（已压缩）】\n{state[f'{prefix}_summary']}"] + recent_speeches

    return speeches


def node_summary(state: DebateState) -> DebateState:
    """
    V1.5 记忆摘要压缩节点
    每轮反思后执行，如果当前轮数超过阈值，就对久远的发言做摘要
    摘要会在后续的反思/驳论/总结阶段代替久远的原始发言，减少 token 消耗
    """
    if not state.get("summary_enabled"):
        return state

    current_round = state["current_round"]
    threshold = state.get("summary_threshold", 2)

    # 轮数没超过阈值，不需要摘要
    if current_round <= threshold:
        return state

    # 需要摘要的发言：立论 + 前 (current_round - threshold) 轮攻辩
    rounds_to_summarize = current_round - threshold

    print(f"[摘要压缩] 第 {current_round} 轮，对前 {rounds_to_summarize} 轮发言做摘要压缩...")

    # 正方摘要
    aff_speeches = [state["affirmative_opening"]] + state["affirmative_clashes"][:rounds_to_summarize]
    aff_debater = _get_debater("affirmative", state["affirmative_stance"])
    aff_summary = aff_debater.summarize(state["topic"], aff_speeches, state.get("affirmative_summary"))
    state["affirmative_summary"] = aff_summary

    # 反方摘要
    neg_speeches = [state["negative_opening"]] + state["negative_clashes"][:rounds_to_summarize]
    neg_debater = _get_debater("negative", state["negative_stance"])
    neg_summary = neg_debater.summarize(state["topic"], neg_speeches, state.get("negative_summary"))
    state["negative_summary"] = neg_summary

    state["full_transcript"].append({
        "speaker": "system",
        "role": "【系统】记忆摘要压缩",
        "content": f"已对前 {rounds_to_summarize} 轮发言完成摘要压缩，后续阶段将使用摘要+最近{threshold}轮原始发言。",
    })

    return state


# ===== 驳论阶段 =====

def node_rebuttal_affirmative(state: DebateState) -> DebateState:
    """正方驳论"""
    # V1.5：如果有摘要，久远部分用摘要代替
    aff_all = _get_side_speeches_for_context(state, "affirmative")
    neg_all = _get_side_speeches_for_context(state, "negative")
    reflection = state.get("affirmative_reflection") if state.get("reflection_enabled") else None

    debater = _get_debater("affirmative", state["affirmative_stance"])
    speech = debater.rebuttal(state["topic"], aff_all, neg_all, reflection)
    state["affirmative_rebuttal"] = speech
    state["current_speaker"] = "negative"
    state["full_transcript"].append({
        "speaker": "affirmative",
        "role": "正方（驳论）",
        "content": speech,
    })
    return state


def node_rebuttal_negative(state: DebateState) -> DebateState:
    """反方驳论"""
    # V1.5：如果有摘要，久远部分用摘要代替
    aff_all = _get_side_speeches_for_context(state, "affirmative")
    neg_all = _get_side_speeches_for_context(state, "negative")
    reflection = state.get("negative_reflection") if state.get("reflection_enabled") else None

    debater = _get_debater("negative", state["negative_stance"])
    speech = debater.rebuttal(state["topic"], neg_all, aff_all, reflection)
    state["negative_rebuttal"] = speech
    state["phase"] = DebatePhase.CLOSING
    state["current_speaker"] = "affirmative"
    state["full_transcript"].append({
        "speaker": "negative",
        "role": "反方（驳论）",
        "content": speech,
    })
    return state


# ===== 总结阶段 =====

def node_closing_affirmative(state: DebateState) -> DebateState:
    """正方总结"""
    # V1.5：如果有摘要，久远部分用摘要代替；总结阶段包含驳论
    aff_all = _get_side_speeches_for_context(state, "affirmative", include_rebuttal=True)
    neg_all = _get_side_speeches_for_context(state, "negative", include_rebuttal=True)

    debater = _get_debater("affirmative", state["affirmative_stance"])
    speech = debater.closing(state["topic"], aff_all, neg_all)
    state["affirmative_closing"] = speech
    state["current_speaker"] = "negative"
    state["full_transcript"].append({
        "speaker": "affirmative",
        "role": "正方（总结陈词）",
        "content": speech,
    })
    return state


def node_closing_negative(state: DebateState) -> DebateState:
    """反方总结"""
    # V1.5：如果有摘要，久远部分用摘要代替；总结阶段包含驳论
    aff_all = _get_side_speeches_for_context(state, "affirmative", include_rebuttal=True)
    neg_all = _get_side_speeches_for_context(state, "negative", include_rebuttal=True)

    debater = _get_debater("negative", state["negative_stance"])
    speech = debater.closing(state["topic"], neg_all, aff_all)
    state["negative_closing"] = speech
    state["phase"] = DebatePhase.JUDGING
    state["full_transcript"].append({
        "speaker": "negative",
        "role": "反方（总结陈词）",
        "content": speech,
    })
    return state


# ===== 评委评分 =====

def node_judge(state: DebateState) -> DebateState:
    """评委评分"""
    aff_full = _format_side_speeches(state, "affirmative")
    neg_full = _format_side_speeches(state, "negative")

    judge = JudgeAgent()
    score = judge.judge(state["topic"], aff_full, neg_full)

    state["judge_score"] = score
    state["winner"] = score.get("winner", "unknown")
    state["phase"] = DebatePhase.DONE
    state["full_transcript"].append({
        "speaker": "judge",
        "role": "评委（评分与点评）",
        "content": f"正方总分：{score.get('affirmative_total', 0)}\n"
                   f"反方总分：{score.get('negative_total', 0)}\n"
                   f"获胜方：{'正方' if score.get('winner') == 'affirmative' else '反方'}\n\n"
                   f"评委点评：\n{score.get('comment', '')}",
    })
    return state


def _format_side_speeches(state: DebateState, side: str) -> str:
    """整理一方的所有发言，供评委评分使用"""
    parts = []
    if side == "affirmative":
        if state["affirmative_opening"]:
            parts.append(f"【立论】\n{state['affirmative_opening']}")
        for i, speech in enumerate(state["affirmative_clashes"]):
            parts.append(f"【第{i+1}轮攻辩】\n{speech}")
        if state["affirmative_rebuttal"]:
            parts.append(f"【驳论】\n{state['affirmative_rebuttal']}")
        if state["affirmative_closing"]:
            parts.append(f"【总结】\n{state['affirmative_closing']}")
    else:
        if state["negative_opening"]:
            parts.append(f"【立论】\n{state['negative_opening']}")
        for i, speech in enumerate(state["negative_clashes"]):
            parts.append(f"【第{i+1}轮攻辩】\n{speech}")
        if state["negative_rebuttal"]:
            parts.append(f"【驳论】\n{state['negative_rebuttal']}")
        if state["negative_closing"]:
            parts.append(f"【总结】\n{state['negative_closing']}")
    return "\n\n".join(parts)
