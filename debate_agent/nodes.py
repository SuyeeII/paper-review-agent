"""论文多视角审稿助手节点定义（多Agent并行评审架构）
4个维度审稿人并行评审 → 自我反思修正 → 主编汇总
"""
from typing import Optional
import re
from debate_agent.state import ReviewState, ReviewPhase
from debate_agent.llm import get_llm
from debate_agent.prompts import (
    get_paper_structure_prompt,
    get_innovation_review_prompt,
    get_methodology_review_prompt,
    get_experiment_review_prompt,
    get_writing_review_prompt,
    get_reflection_prompt,
    get_editor_summary_prompt,
)


# ===== 后处理：过滤越界的主要缺陷（第二层防护，不依赖LLM，100%可靠）=====

# 每个维度的越界关键词：如果某条主要缺陷包含这些关键词，就删掉
# 注意：每个维度都包含公式/图/表格相关关键词，因为PDF解析不完整，无法准确判断这些内容
CROSS_DIMENSION_KEYWORDS = {
    "innovation": [  # 创新性审稿人不能评价的内容
        "代码", "可复现", "复现", "对比实验", "消融实验", "消融",
        "收敛性证明", "理论推导", "数学推导", "数学依据", "基线方法", "基线",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "语言", "语法", "拼写", "图表规范", "参考文献格式",
        # 创新性审稿人不能评价方法论维度的问题
        "收敛性", "稳定性", "理论基础", "方法假设", "方法局限性", "技术路线",
        # PDF解析限制：公式/图/表格无法准确判断，所有维度都过滤
        "公式", "推导", "数学依据", "数学证明", "图1", "图2", "图3", "图4", "图5",
        "表1", "表2", "表3", "表4", "表5", "图表", "坐标轴", "图例", "表格",
    ],
    "methodology": [  # 方法论审稿人不能评价的内容
        "代码", "可复现", "复现", "对比实验", "消融实验", "消融",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "基线方法", "基线", "语言", "语法", "拼写",
        "创新点", "创新性", "相关工作对比", "技术贡献",
        # PDF解析限制：公式/图/表格无法准确判断，所有维度都过滤
        "公式", "推导", "数学依据", "数学证明", "图1", "图2", "图3", "图4", "图5",
        "表1", "表2", "表3", "表4", "表5", "图表", "坐标轴", "图例", "表格",
    ],
    "experiment": [  # 实验审稿人不能评价的内容
        "创新点", "创新性", "相关工作对比", "技术贡献", "贡献大小",
        "语言", "语法", "拼写", "图表规范", "参考文献格式",
        "理论推导", "数学推导", "收敛性证明", "数学依据",
        # 早期会议论文不公开代码是惯例，不应该列为缺陷（代码铁律）
        "未公开代码", "代码公开",
        # PDF解析限制：公式/图/表格无法准确判断，所有维度都过滤
        "公式", "推导", "数学依据", "数学证明", "图1", "图2", "图3", "图4", "图5",
        "表1", "表2", "表3", "表4", "表5", "图表", "坐标轴", "图例", "表格",
    ],
    "writing": [  # 写作审稿人不能评价的内容
        "对比实验", "消融实验", "消融", "基线方法", "基线",
        "代码公开", "未公开代码", "代码", "可复现", "复现",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "创新点", "创新性", "理论推导", "数学推导", "收敛性证明", "数学依据",
        "方法假设", "技术路线",
        # PDF解析限制：公式/推导/数学依据无法准确判断，写作审稿人也过滤
        # 注意：写作审稿人可以评价"图表的文字描述是否清晰"，但不能评价图表标题是否准确（因为看不到图表内容）
        "公式", "推导", "数学依据", "数学证明",
        "图表标题", "图标题", "表标题",
    ],
}


def _filter_section_items(content: str, keywords: list) -> tuple:
    """
    过滤一个section（主要缺陷/具体修改建议）里的条目
    返回 (过滤后的内容, 删除的条目数)
    """
    # 按编号拆分成条目（匹配 "1. " "2. " 等）
    items = re.split(r'(?=\d+\.\s)', content.strip())
    items = [item.strip() for item in items if item.strip()]

    # 过滤掉包含越界关键词的条目
    filtered_items = []
    removed_count = 0
    for item in items:
        if any(kw in item for kw in keywords):
            removed_count += 1
            continue
        filtered_items.append(item)

    # 重新编号
    renumbered_items = []
    for i, item in enumerate(filtered_items, 1):
        item = re.sub(r'^\d+\.\s', f'{i}. ', item)
        renumbered_items.append(item)

    new_content = '\n'.join(renumbered_items)
    if not new_content:
        new_content = "（本维度未发现明显问题）"

    return new_content, removed_count


def _filter_deduction_note(content: str, keywords: list, dimension: str) -> str:
    """
    过滤扣分说明里的越界关键词
    如果扣分说明包含越界关键词，就替换成更通用的表述
    """
    if any(kw in content for kw in keywords):
        # 替换成通用表述
        generic_notes = {
            "innovation": "扣分主要因为创新性描述不够具体、相关工作对比不足、技术贡献有限。",
            "methodology": "扣分主要因为理论推导不够严谨、方法假设不够清晰、技术路线描述不够明确。",
            "experiment": "扣分主要因为实验设计不够充分、结果可靠性有待提升、可复现性细节不够完善。",
            "writing": "扣分主要因为语言表达不够准确、结构不够清晰、参考文献格式不够规范。",
        }
        return generic_notes.get(dimension, content)
    return content


def filter_cross_dimension_issues(review_text: str, dimension: str) -> str:
    """
    后处理过滤：删掉审稿人输出中越界的主要缺陷、具体修改建议，修正扣分说明
    这是第二层防护，不依赖LLM遵守prompt，100%可靠
    """
    if dimension not in CROSS_DIMENSION_KEYWORDS:
        return review_text

    keywords = CROSS_DIMENSION_KEYWORDS[dimension]
    total_removed = 0

    # ===== 1. 过滤"主要缺陷"部分 =====
    pattern_defects = r'(\*\*主要缺陷\*\*[：:]\s*\n)(.*?)(\n\*\*[^\*]+\*\*[：:])'
    match = re.search(pattern_defects, review_text, re.DOTALL)
    if match:
        prefix = match.group(1)
        defects_content = match.group(2)
        suffix = match.group(3)
        new_defects, removed = _filter_section_items(defects_content, keywords)
        total_removed += removed
        review_text = (
            review_text[:match.start()]
            + prefix + new_defects + suffix
            + review_text[match.end():]
        )

    # ===== 2. 过滤"具体修改建议"部分 =====
    pattern_suggestions = r'(\*\*具体修改建议\*\*[：:]\s*\n)(.*?)(\n\*\*[^\*]+\*\*[：:])'
    match = re.search(pattern_suggestions, review_text, re.DOTALL)
    if match:
        prefix = match.group(1)
        suggestions_content = match.group(2)
        suffix = match.group(3)
        new_suggestions, removed = _filter_section_items(suggestions_content, keywords)
        total_removed += removed
        review_text = (
            review_text[:match.start()]
            + prefix + new_suggestions + suffix
            + review_text[match.end():]
        )

    # ===== 3. 修正"扣分说明"部分 =====
    pattern_deduction = r'(\*\*扣分说明\*\*[：:]\s*)(.*?)(\n|$)'
    match = re.search(pattern_deduction, review_text, re.DOTALL)
    if match:
        prefix = match.group(1)
        deduction_content = match.group(2)
        suffix = match.group(3)
        new_deduction = _filter_deduction_note(deduction_content, keywords, dimension)
        if new_deduction != deduction_content:
            total_removed += 1
            review_text = (
                review_text[:match.start()]
                + prefix + new_deduction + suffix
                + review_text[match.end():]
            )

    if total_removed > 0:
        print(f"[后处理过滤] {dimension}审稿人：删除/修正了{total_removed}处越界内容（主要缺陷+修改建议+扣分说明）")

    return review_text


# ===== 论文结构解析节点 =====

def node_paper_structure(state: ReviewState) -> ReviewState:
    """解析论文结构，提取各部分核心内容，为分部分评审提供依据"""
    llm = get_llm()
    prompt = get_paper_structure_prompt(state["topic"])
    structure = llm.chat(prompt, system_prompt="你是一位学术论文结构分析专家，擅长解析论文的各个部分并提取核心内容。", temperature=0.2)
    state["paper_structure"] = structure
    state["full_transcript"].append({
        "reviewer": "structure",
        "role": "论文结构解析",
        "content": structure,
    })
    print("[结构解析] 论文结构解析完成")
    return state


# ===== 4个维度审稿节点（并行执行）=====

def node_innovation_review(state: ReviewState) -> ReviewState:
    """创新性审稿人"""
    llm = get_llm()
    prompt = get_innovation_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文创新性审稿人，擅长评估论文的创新点、研究贡献和相关工作对比。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "innovation")
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
    prompt = get_methodology_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文方法论审稿人，擅长评估研究方法的合理性、理论推导的严谨性和技术路线的清晰度。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "methodology")
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
    prompt = get_experiment_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文实验审稿人，擅长评估实验设计的科学性、结果的可靠性，以及实验的可复现性。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "experiment")
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
    prompt = get_writing_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文写作审稿人，擅长评估论文结构、语言表达、图表规范和参考文献完整性。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "writing")
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
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "innovation")
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
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "methodology")
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
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "experiment")
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
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "writing")
    state["writing_final"] = final
    state["full_transcript"].append({
        "reviewer": "writing",
        "role": "写作审稿人（反思修正后）",
        "content": final,
    })
    print("[反思] 写作审稿反思修正完成")
    return state


# ===== 主编汇总节点 =====

def _fix_editor_total_score(summary: str) -> str:
    """
    后处理：自动计算主编综合评分，替换掉LLM可能算错的加法
    综合评分 = 创新性 + 方法论 + 实验 + 写作 四个维度评分之和
    """
    # 匹配各维度评分表格里的分数
    # 格式：| 创新性 | X | ... |
    # 格式：| 方法论 | X | ... |
    # 格式：| 实验可靠性与可复现性 | X | ... |
    # 格式：| 写作表达 | X | ... |
    pattern_innovation = r'\|\s*创新性\s*\|\s*(\d+)\s*\|'
    pattern_methodology = r'\|\s*方法论\s*\|\s*(\d+)\s*\|'
    pattern_experiment = r'\|\s*实验可靠性与可复现性\s*\|\s*(\d+)\s*\|'
    pattern_writing = r'\|\s*写作表达\s*\|\s*(\d+)\s*\|'

    match_innovation = re.search(pattern_innovation, summary)
    match_methodology = re.search(pattern_methodology, summary)
    match_experiment = re.search(pattern_experiment, summary)
    match_writing = re.search(pattern_writing, summary)

    if not all([match_innovation, match_methodology, match_experiment, match_writing]):
        return summary  # 匹配不到就不修改

    innovation_score = int(match_innovation.group(1))
    methodology_score = int(match_methodology.group(1))
    experiment_score = int(match_experiment.group(1))
    writing_score = int(match_writing.group(1))

    total_score = innovation_score + methodology_score + experiment_score + writing_score

    # 替换综合评分
    # 格式：| **综合评分** | **X/40** | |
    pattern_total = r'(\|\s*\*\*综合评分\*\*\s*\|\s*\*\*)\d+(/40\*\*\s*\|)'
    match_total = re.search(pattern_total, summary)
    if match_total:
        new_summary = summary[:match_total.start()] + match_total.group(1) + str(total_score) + match_total.group(2) + summary[match_total.end():]
        print(f"[后处理] 主编综合评分修正：LLM算错了，正确应为{total_score}/40（{innovation_score}+{methodology_score}+{experiment_score}+{writing_score}）")
        return new_summary

    return summary


def _fix_editor_defects_format(summary: str) -> str:
    """
    后处理：去掉主编主要缺陷里的加粗格式，保持格式统一
    有些审稿人的主要缺陷里有加粗标题（如**实验部分文字描述不足**：...），
    主编汇总时直接复制会导致有的条目加粗有的不加粗，格式不一致。
    """
    # 匹配"## 四、主要缺陷"到"## 五、"之间的内容
    pattern = r'(## 四、主要缺陷\n.*?\n)(.*?)(\n## 五、)'
    match = re.search(pattern, summary, re.DOTALL)
    if not match:
        return summary

    prefix = match.group(1)
    defects_content = match.group(2)
    suffix = match.group(3)

    # 去掉加粗格式：把**...**替换成...
    # 但要注意不要去掉"（创新性审稿人）"这种括号里的内容
    new_defects_content = re.sub(r'\*\*(.+?)\*\*', r'\1', defects_content)

    if new_defects_content == defects_content:
        return summary  # 没有加粗，不需要修改

    new_summary = summary[:match.start()] + prefix + new_defects_content + suffix + summary[match.end():]
    print("[后处理] 主编主要缺陷格式修正：去掉了加粗，保持格式统一")
    return new_summary


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
    # 后处理：自动计算综合评分，替换掉LLM可能算错的加法
    summary = _fix_editor_total_score(summary)
    # 后处理：去掉主要缺陷里的加粗格式，保持格式统一
    summary = _fix_editor_defects_format(summary)
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
