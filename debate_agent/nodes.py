"""LangGraph 图节点函数
每个节点接收 state，处理后返回更新后的 state
"""
from typing import Dict, Any
from .state import DebateState, DebatePhase
from .debater import DebaterAgent
from .judge import JudgeAgent


# ===== 立论阶段 =====

def node_opening_affirmative(state: DebateState) -> DebateState:
    """正方立论"""
    debater = DebaterAgent("affirmative", state["affirmative_stance"])
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
    debater = DebaterAgent("negative", state["negative_stance"])
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

    debater = DebaterAgent("affirmative", state["affirmative_stance"])
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

    debater = DebaterAgent("negative", state["negative_stance"])
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

    # 收集双方到目前为止的所有发言
    aff_speeches = [state["affirmative_opening"]] + state["affirmative_clashes"]
    neg_speeches = [state["negative_opening"]] + state["negative_clashes"]

    # 正方反思（参考自己的所有发言 + 反方的所有发言）
    aff_debater = DebaterAgent("affirmative", state["affirmative_stance"])
    aff_reflection = aff_debater.reflect(state["topic"], aff_speeches, neg_speeches)
    state["affirmative_reflection"] = aff_reflection

    # 反方反思（参考自己的所有发言 + 正方的所有发言）
    neg_debater = DebaterAgent("negative", state["negative_stance"])
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
    """
    if state["current_round"] >= state["max_rounds"]:
        # 攻辩结束，进入驳论
        return "rebuttal_affirmative"
    else:
        # 继续下一轮攻辩，轮数+1
        state["current_round"] += 1
        return "clash_affirmative"


# ===== 驳论阶段 =====

def node_rebuttal_affirmative(state: DebateState) -> DebateState:
    """正方驳论"""
    aff_all = [state["affirmative_opening"]] + state["affirmative_clashes"]
    neg_all = [state["negative_opening"]] + state["negative_clashes"]
    reflection = state.get("affirmative_reflection") if state.get("reflection_enabled") else None

    debater = DebaterAgent("affirmative", state["affirmative_stance"])
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
    aff_all = [state["affirmative_opening"]] + state["affirmative_clashes"]
    neg_all = [state["negative_opening"]] + state["negative_clashes"]
    reflection = state.get("negative_reflection") if state.get("reflection_enabled") else None

    debater = DebaterAgent("negative", state["negative_stance"])
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
    aff_all = [state["affirmative_opening"]] + state["affirmative_clashes"] + [state["affirmative_rebuttal"]]
    neg_all = [state["negative_opening"]] + state["negative_clashes"] + [state["negative_rebuttal"]]

    debater = DebaterAgent("affirmative", state["affirmative_stance"])
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
    aff_all = [state["affirmative_opening"]] + state["affirmative_clashes"] + [state["affirmative_rebuttal"]]
    neg_all = [state["negative_opening"]] + state["negative_clashes"] + [state["negative_rebuttal"]]

    debater = DebaterAgent("negative", state["negative_stance"])
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
