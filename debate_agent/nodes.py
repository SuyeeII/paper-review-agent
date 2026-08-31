"""论文多视角审稿助手节点定义（多Agent并行评审架构）
4个维度审稿人并行评审 → 自我反思修正 → 主编汇总
"""
from typing import Optional
from debate_agent.state import ReviewState, ReviewPhase
from debate_agent.llm import get_llm
from debate_agent.prompts import (
    get_innovation_review_prompt,
    get_methodology_review_prompt,
    get_experiment_review_prompt,
    get_writing_review_prompt,
    get_reflection_prompt,
    get_editor_summary_prompt,
)


# ===== 4个维度审稿节点（并行执行）=====

def node_innovation_review(state: ReviewState) -> ReviewState:
    """创新性审稿人"""
    llm = get_llm()
    prompt = get_innovation_review_prompt(state["topic"])
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文创新性审稿人，擅长评估论文的创新点、研究贡献和相关工作对比。")
    state["innovation_review"] = review
    state["full_transcript"].append({
        "reviewer": "innovation",
        "role": "创新性审稿人（初审）",
        "content": review,
    })
    print("[审稿] 创新性审稿完成")
    return state


def node_methodology_review(state: ReviewState) -> ReviewState:
    """方法论审稿人"""
    llm = get_llm()
    prompt = get_methodology_review_prompt(state["topic"])
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文方法论审稿人，擅长评估研究方法的合理性、理论推导的严谨性和技术路线的清晰度。")
    state["methodology_review"] = review
    state["full_transcript"].append({
        "reviewer": "methodology",
        "role": "方法论审稿人（初审）",
        "content": review,
    })
    print("[审稿] 方法论审稿完成")
    return state


def node_experiment_review(state: ReviewState) -> ReviewState:
    """实验审稿人（含可复现性评估）"""
    llm = get_llm()
    prompt = get_experiment_review_prompt(state["topic"])
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文实验审稿人，擅长评估实验设计的科学性、结果的可靠性，以及实验的可复现性。")
    state["experiment_review"] = review
    state["full_transcript"].append({
        "reviewer": "experiment",
        "role": "实验审稿人（初审，含可复现性评估）",
        "content": review,
    })
    print("[审稿] 实验审稿完成")
    return state


def node_writing_review(state: ReviewState) -> ReviewState:
    """写作审稿人"""
    llm = get_llm()
    prompt = get_writing_review_prompt(state["topic"])
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文写作审稿人，擅长评估论文结构、语言表达、图表规范和参考文献完整性。")
    state["writing_review"] = review
    state["full_transcript"].append({
        "reviewer": "writing",
        "role": "写作审稿人（初审）",
        "content": review,
    })
    print("[审稿] 写作审稿完成")
    return state


# ===== 4个维度反思修正节点（并行执行，仅当reflection_enabled时）=====

def node_innovation_reflection(state: ReviewState) -> ReviewState:
    """创新性审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("创新性", state["topic"], state["innovation_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文创新性审稿人，正在对自己的初审意见进行自我反思和修正。")
    state["innovation_final"] = final
    state["full_transcript"].append({
        "reviewer": "innovation",
        "role": "创新性审稿人（反思修正后）",
        "content": final,
    })
    print("[反思] 创新性审稿反思修正完成")
    return state


def node_methodology_reflection(state: ReviewState) -> ReviewState:
    """方法论审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("方法论", state["topic"], state["methodology_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文方法论审稿人，正在对自己的初审意见进行自我反思和修正。")
    state["methodology_final"] = final
    state["full_transcript"].append({
        "reviewer": "methodology",
        "role": "方法论审稿人（反思修正后）",
        "content": final,
    })
    print("[反思] 方法论审稿反思修正完成")
    return state


def node_experiment_reflection(state: ReviewState) -> ReviewState:
    """实验审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("实验可靠性与可复现性", state["topic"], state["experiment_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文实验审稿人，正在对自己的初审意见进行自我反思和修正。")
    state["experiment_final"] = final
    state["full_transcript"].append({
        "reviewer": "experiment",
        "role": "实验审稿人（反思修正后）",
        "content": final,
    })
    print("[反思] 实验审稿反思修正完成")
    return state


def node_writing_reflection(state: ReviewState) -> ReviewState:
    """写作审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("写作表达", state["topic"], state["writing_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文写作审稿人，正在对自己的初审意见进行自我反思和修正。")
    state["writing_final"] = final
    state["full_transcript"].append({
        "reviewer": "writing",
        "role": "写作审稿人（反思修正后）",
        "content": final,
    })
    print("[反思] 写作审稿反思修正完成")
    return state


# ===== 主编汇总节点 =====

def node_editor_summary(state: ReviewState) -> ReviewState:
    """主编汇总：汇总4份审稿意见，给出综合审稿报告"""
    # 如果启用了反思，用修正后的意见；否则用初审意见
    innovation = state.get("innovation_final") or state.get("innovation_review") or ""
    methodology = state.get("methodology_final") or state.get("methodology_review") or ""
    experiment = state.get("experiment_final") or state.get("experiment_review") or ""
    writing = state.get("writing_final") or state.get("writing_review") or ""

    llm = get_llm()
    prompt = get_editor_summary_prompt(state["topic"], innovation, methodology, experiment, writing)
    summary = llm.chat(prompt, system_prompt="你是一位资深的学术期刊领域主编（Area Chair），负责汇总多位审稿人的意见，给出最终的综合审稿报告和录用决定。", temperature=0.3)
    state["editor_summary"] = summary
    state["phase"] = ReviewPhase.DONE
    state["full_transcript"].append({
        "reviewer": "editor",
        "role": "主编综合审稿报告",
        "content": summary,
    })
    print("[汇总] 主编综合审稿报告完成")
    return state


# ===== 路由函数 =====

def route_after_review(state: ReviewState) -> str:
    """
    4个审稿人完成后，决定下一步：
    - 如果启用了自我反思，进入反思修正阶段
    - 否则直接进入主编汇总
    """
    if state.get("reflection_enabled", True):
        return "reflection"
    else:
        return "summary"
