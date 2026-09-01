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
# 注意：这是"黑名单"方式，列出每个维度绝对不能评价的所有关键词
# 每个维度都包含公式/图/表格相关关键词，因为PDF解析不完整，无法准确判断这些内容
CROSS_DIMENSION_KEYWORDS = {
    "innovation": [  # 创新性审稿人不能评价的内容
        # ===== 实验维度所有问题 =====
        "实验设计", "实验结果", "实验设置", "实验部分",
        "参数设置", "参数取值", "参数范围", "实验参数", "超参数",
        "对比实验", "消融实验", "消融", "基线方法", "基线",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "可复现", "复现", "代码",
        # ===== 方法论维度所有问题 =====
        "方法合理性", "技术路线", "理论依据",
        "收敛性", "稳定性", "理论基础", "方法假设", "方法局限性", "局限性", "算法的局限性",
        "理论推导", "数学推导", "数学依据", "收敛性证明",
        # ===== 写作维度所有问题 =====
        "语言", "语法", "拼写", "图表规范", "参考文献",
        # ===== PDF解析限制：公式/图/表格无法准确判断 =====
        "公式", "推导", "数学证明",
        "图1", "图2", "图3", "图4", "图5", "图6", "图7", "图8", "图9", "图10",
        "表1", "表2", "表3", "表4", "表5", "表6", "表7", "表8", "表9", "表10",
        "图表", "坐标轴", "图例", "表格",
    ],
    "methodology": [  # 方法论审稿人不能评价的内容
        # ===== 实验维度所有问题 =====
        "实验设计", "实验结果", "实验设置", "实验部分",
        "参数设置", "参数取值", "参数范围", "实验参数", "超参数",
        "对比实验", "消融实验", "消融", "基线方法", "基线",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "可复现", "复现", "代码",
        # ===== 创新性维度所有问题 =====
        "创新点", "创新性", "相关工作对比", "技术贡献", "贡献大小", "研究时效性",
        "对比分析", "与现有算法对比", "与现有PSO",
        # ===== 写作维度所有问题 =====
        "语言", "语法", "拼写", "图表规范", "参考文献",
        # ===== PDF解析限制：公式/图/表格无法准确判断 =====
        # 注意：方法论审稿人可以评价方法合理性/技术路线/理论依据，但不能评价公式推导本身
        "公式", "推导", "数学证明", "理论推导", "数学推导", "数学依据", "收敛性证明",
        "图1", "图2", "图3", "图4", "图5", "图6", "图7", "图8", "图9", "图10",
        "表1", "表2", "表3", "表4", "表5", "表6", "表7", "表8", "表9", "表10",
        "图表", "坐标轴", "图例", "表格",
    ],
    "experiment": [  # 实验审稿人不能评价的内容
        # ===== 创新性维度所有问题 =====
        "创新点", "创新性", "相关工作对比", "技术贡献", "贡献大小", "研究时效性",
        # ===== 方法论维度所有问题 =====
        "方法合理性", "技术路线", "理论依据",
        "收敛性", "稳定性", "理论基础", "方法假设", "方法局限性",
        # ===== 写作维度所有问题 =====
        "语言", "语法", "拼写", "图表规范", "参考文献",
        # ===== 代码铁律：论文里看不到代码相关情况，不管什么代码都不该评价 =====
        "代码",
        # ===== PDF解析限制：公式/图/表格无法准确判断 =====
        "公式", "推导", "数学证明", "理论推导", "数学推导", "数学依据", "收敛性证明",
        "图1", "图2", "图3", "图4", "图5", "图6", "图7", "图8", "图9", "图10",
        "表1", "表2", "表3", "表4", "表5", "表6", "表7", "表8", "表9", "表10",
        "图表", "坐标轴", "图例", "表格",
    ],
    "writing": [  # 写作审稿人不能评价的内容
        # ===== 实验维度所有问题 =====
        "实验设计", "实验结果", "实验设置", "实验部分",
        "参数设置", "参数取值", "参数范围", "实验参数", "超参数",
        "对比实验", "消融实验", "消融", "基线方法", "基线",
        "数据集", "评价指标", "统计显著性", "标准差", "方差",
        "可复现", "复现", "代码",
        # ===== 创新性维度所有问题 =====
        "创新点", "创新性", "相关工作对比", "技术贡献", "贡献大小", "研究时效性",
        # ===== 方法论维度所有问题 =====
        "方法合理性", "技术路线", "理论依据",
        "收敛性", "稳定性", "理论基础", "方法假设", "方法局限性",
        "理论推导", "数学推导", "数学依据", "收敛性证明",
        # ===== PDF解析限制：公式/图/表格/参考文献无法准确判断 =====
        "公式", "推导", "数学证明",
        "图表", "图表标题", "图标题", "表标题",
        "图1", "图2", "图3", "图4", "图5", "图6", "图7", "图8", "图9", "图10",
        "表1", "表2", "表3", "表4", "表5", "表6", "表7", "表8", "表9", "表10",
        "坐标轴", "图例", "表格",
        "参考文献", "引用格式", "文献格式", "引用不规范",
        # ===== 写作审稿人只评价文字表达，不评价内容本身 =====
        "不够充分", "不够深入", "不够详细", "不够具体", "过于简略", "内容不充分", "描述不详细", "分析不深入", "描述不够具体",
        "相关工作介绍", "算法描述", "计算过程", "动态半径设置策略的计算过程",
        # 注意：写作审稿人可以评价语言/语法/拼写/结构逻辑/参考文献/文字表达是否清晰，这些不过滤
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
            "experiment": "扣分主要因为论证不够充分、证据可靠性有待提升、结论支撑不够完善。",
            "writing": "扣分主要因为语言表达不够准确、结构不够清晰、文字表达不够简洁。",
        }
        return generic_notes.get(dimension, content)
    return content


def filter_cross_dimension_issues(review_text: str, dimension: str) -> str:
    """
    后处理过滤：删掉审稿人输出中越界的主要缺陷、具体修改建议，修正扣分说明
    这是第二层防护，不依赖LLM遵守prompt，100%可靠
    注意：会处理所有出现的"主要缺陷""具体修改建议""扣分说明"部分（包括初审和反思修正后的部分）
    """
    if dimension not in CROSS_DIMENSION_KEYWORDS:
        return review_text

    keywords = CROSS_DIMENSION_KEYWORDS[dimension]
    total_removed = 0

    # ===== 1. 过滤所有"主要缺陷"部分 =====
    # 先收集所有匹配，再倒序处理，避免while循环无限循环（如果过滤后内容没变，会一直匹配同一个）
    pattern_defects = r'(\*\*主要缺陷\*\*[：:]\s*\n)(.*?)(\n\*\*[^\*]+\*\*[：:])'
    matches_defects = list(re.finditer(pattern_defects, review_text, re.DOTALL))
    for match in reversed(matches_defects):
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

    # ===== 2. 过滤所有"具体修改建议"部分 =====
    pattern_suggestions = r'(\*\*具体修改建议\*\*[：:]\s*\n)(.*?)(\n\*\*[^\*]+\*\*[：:])'
    matches_suggestions = list(re.finditer(pattern_suggestions, review_text, re.DOTALL))
    for match in reversed(matches_suggestions):
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

    # ===== 3. 修正所有"扣分说明"部分 =====
    pattern_deduction = r'(\*\*扣分说明\*\*[：:]\s*)(.*?)(\n|$)'
    matches_deduction = list(re.finditer(pattern_deduction, review_text, re.DOTALL))
    for match in reversed(matches_deduction):
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
        print(f"[后处理过滤] {dimension}审稿人：删除/修正了{total_removed}处越界内容（主要缺陷+修改建议+扣分说明，包含所有出现的部分）")

    return review_text


# ===== 论文结构解析节点 =====

def node_paper_structure(state: ReviewState) -> ReviewState:
    """解析论文结构，提取各部分核心内容，为分部分评审提供依据"""
    llm = get_llm()
    prompt = get_paper_structure_prompt(state["topic"])
    structure = llm.chat(prompt, system_prompt="你是一位学术论文结构分析专家，擅长解析论文的各个部分并提取核心内容。", temperature=0.2)
    print("[结构解析] 论文结构解析完成")
    # 只返回修改的字段，full_transcript返回要追加的内容（用operator.add合并）
    return {
        "paper_structure": structure,
        "full_transcript": [{
            "reviewer": "structure",
            "role": "论文结构解析",
            "content": structure,
        }],
    }


# ===== 4个维度审稿节点（并行执行）=====

def node_innovation_review(state: ReviewState) -> ReviewState:
    """创新性审稿人"""
    llm = get_llm()
    prompt = get_innovation_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文创新性审稿人，擅长评估论文的创新点、研究贡献和相关工作对比。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "innovation")
    print("[审稿] 创新性审稿完成")
    return {
        "innovation_review": review,
        "full_transcript": [{
            "reviewer": "innovation",
            "role": "创新性审稿人（初审）",
            "content": review,
        }],
    }


def node_methodology_review(state: ReviewState) -> ReviewState:
    """方法论审稿人"""
    llm = get_llm()
    prompt = get_methodology_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文方法论审稿人，擅长评估研究方法的合理性、理论推导的严谨性和技术路线的清晰度。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "methodology")
    print("[审稿] 方法论审稿完成")
    return {
        "methodology_review": review,
        "full_transcript": [{
            "reviewer": "methodology",
            "role": "方法论审稿人（初审）",
            "content": review,
        }],
    }


def node_experiment_review(state: ReviewState) -> ReviewState:
    """论证与证据审稿人"""
    llm = get_llm()
    prompt = get_experiment_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文论证与证据审稿人，擅长评估论证是否充分、证据是否可靠、结论是否有充分支撑（适用于所有学科，包括理工科的实验数据、文科的案例分析、理论研究的逻辑推导等）。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "experiment")
    print("[审稿] 论证与证据审稿完成")
    return {
        "experiment_review": review,
        "full_transcript": [{
            "reviewer": "experiment",
            "role": "论证与证据审稿人（初审）",
            "content": review,
        }],
    }


def node_writing_review(state: ReviewState) -> ReviewState:
    """写作审稿人"""
    llm = get_llm()
    prompt = get_writing_review_prompt(state["topic"], state.get("paper_structure"))
    review = llm.chat(prompt, system_prompt="你是一位严谨的学术论文写作审稿人，擅长评估论文结构、语言表达、图表规范和参考文献完整性。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    review = filter_cross_dimension_issues(review, "writing")
    print("[审稿] 写作审稿完成")
    return {
        "writing_review": review,
        "full_transcript": [{
            "reviewer": "writing",
            "role": "写作审稿人（初审）",
            "content": review,
        }],
    }


# ===== 4个维度反思修正节点（并行执行，仅当reflection_enabled时）=====

def node_innovation_reflection(state: ReviewState) -> ReviewState:
    """创新性审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("创新性", state["topic"], state["innovation_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文创新性审稿人，正在对自己的初审意见进行自我反思和修正。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "innovation")
    print("[反思] 创新性审稿反思修正完成")
    return {
        "innovation_final": final,
        "full_transcript": [{
            "reviewer": "innovation",
            "role": "创新性审稿人（反思修正后）",
            "content": final,
        }],
    }


def node_methodology_reflection(state: ReviewState) -> ReviewState:
    """方法论审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("方法论", state["topic"], state["methodology_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文方法论审稿人，正在对自己的初审意见进行自我反思和修正。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "methodology")
    print("[反思] 方法论审稿反思修正完成")
    return {
        "methodology_final": final,
        "full_transcript": [{
            "reviewer": "methodology",
            "role": "方法论审稿人（反思修正后）",
            "content": final,
        }],
    }


def node_experiment_reflection(state: ReviewState) -> ReviewState:
    """实验审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("论证与证据", state["topic"], state["experiment_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文实验审稿人，正在对自己的初审意见进行自我反思和修正。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "experiment")
    print("[反思] 实验审稿反思修正完成")
    return {
        "experiment_final": final,
        "full_transcript": [{
            "reviewer": "experiment",
            "role": "实验审稿人（反思修正后）",
            "content": final,
        }],
    }


def node_writing_reflection(state: ReviewState) -> ReviewState:
    """写作审稿人自我反思修正"""
    llm = get_llm()
    prompt = get_reflection_prompt("写作表达", state["topic"], state["writing_review"])
    final = llm.chat(prompt, system_prompt="你是一位严谨的学术论文写作审稿人，正在对自己的初审意见进行自我反思和修正。")
    # 后处理过滤：删掉越界的主要缺陷（第二层防护）
    final = filter_cross_dimension_issues(final, "writing")
    print("[反思] 写作审稿反思修正完成")
    return {
        "writing_final": final,
        "full_transcript": [{
            "reviewer": "writing",
            "role": "写作审稿人（反思修正后）",
            "content": final,
        }],
    }


# ===== 主编汇总节点 =====

def _fix_editor_total_score(summary: str) -> str:
    """
    后处理：自动计算主编综合评分，替换掉LLM可能算错的加法
    综合评分 = 创新性 + 方法论 + 实验 + 写作 四个维度评分之和
    """
    # 匹配各维度评分表格里的分数
    # 格式：| 创新性 | X | ... |
    # 格式：| 方法论 | X | ... |
    # 格式：| 论证与证据 | X | ... |
    # 格式：| 写作表达 | X | ... |
    pattern_innovation = r'\|\s*创新性\s*\|\s*(\d+)\s*\|'
    pattern_methodology = r'\|\s*方法论\s*\|\s*(\d+)\s*\|'
    pattern_experiment = r'\|\s*论证与证据\s*\|\s*(\d+)\s*\|'
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


def _deduplicate_items(items: list, threshold: float = 0.5) -> tuple:
    """
    对条目列表进行去重，基于关键词相似度
    返回 (去重后的列表, 去除的重复条目数量)
    """
    if len(items) <= 1:
        return items, 0

    # 简单的中文停用词
    stop_words = set("的了是在和有与及对这那也都就而但如果因为所以我们你们他们它们一个一些一种这些那些可以能够需要应该通过进行基于以及等等".split())

    def extract_keywords(text: str) -> set:
        """提取关键词：去掉编号、标点、停用词，保留2字以上的词"""
        # 去掉编号
        text = re.sub(r'^\d+\.\s*', '', text)
        # 去掉加粗标记
        text = text.replace('**', '')
        # 去掉标点和特殊字符
        text = re.sub(r'[，。、；：！？""''（）【】《》\s\.\,\;\:\!\?\(\)\[\]<>]', ' ', text)
        # 简单分词：按空格分割，然后过滤
        words = text.split()
        keywords = set()
        for w in words:
            if len(w) >= 2 and w not in stop_words:
                keywords.add(w)
        # 对于中文连续文本，再做简单的2-gram切分
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        for seg in chinese_chars:
            for i in range(len(seg) - 1):
                bigram = seg[i:i+2]
                if bigram not in stop_words:
                    keywords.add(bigram)
        return keywords

    unique_items = []
    removed_count = 0
    item_keywords = []

    for item in items:
        kw = extract_keywords(item)
        is_duplicate = False
        for i, existing_kw in enumerate(item_keywords):
            if len(kw) == 0 or len(existing_kw) == 0:
                continue
            # 计算Jaccard相似度
            intersection = len(kw & existing_kw)
            union = len(kw | existing_kw)
            similarity = intersection / union if union > 0 else 0
            if similarity >= threshold:
                is_duplicate = True
                removed_count += 1
                break
        if not is_duplicate:
            unique_items.append(item)
            item_keywords.append(kw)

    return unique_items, removed_count


def _filter_editor_defects_by_dimension(items: list) -> tuple:
    """
    主编主要缺陷的维度归属校验
    检查每条缺陷是否明显属于其他维度，如果是就过滤掉
    返回 (过滤后的列表, 去除的越界条目数量)
    """
    # 各维度的核心关键词（用于判断一条缺陷主要属于哪个维度）
    dimension_keywords = {
        "innovation": ["创新点", "创新性", "相关工作", "技术贡献", "研究时效", "与现有", "对比分析"],
        "methodology": ["方法合理", "技术路线", "理论依据", "理论基础", "方法假设", "方法局限", "收敛性", "稳定性", "参数设置", "算法原理"],
        "experiment": ["实验设计", "实验结果", "实验设置", "对比实验", "消融实验", "数据集", "评价指标", "统计显著", "论证", "证据", "数据支撑"],
        "writing": ["语言", "语法", "拼写", "表达", "结构清晰", "文字", "术语", "参考文献"],
    }

    filtered_items = []
    removed_count = 0

    for item in items:
        # 提取来源标注（如"（创新性审稿人）"）
        source_match = re.search(r'[（(](\w+?)审稿人[）)]', item)
        if not source_match:
            # 没有来源标注，保留
            filtered_items.append(item)
            continue

        source = source_match.group(1)
        # 映射来源到维度key
        source_dim_map = {
            "创新性": "innovation",
            "方法论": "methodology",
            "论证与证据": "experiment",
            "写作表达": "writing",
        }
        source_dim = source_dim_map.get(source)
        if not source_dim:
            filtered_items.append(item)
            continue

        # 检查这条缺陷是否主要属于其他维度
        other_dim_scores = {}
        for dim, keywords in dimension_keywords.items():
            if dim == source_dim:
                continue
            score = sum(1 for kw in keywords if kw in item)
            other_dim_scores[dim] = score

        # 如果属于其他维度的关键词数量明显多于来源维度，就过滤掉
        source_score = sum(1 for kw in dimension_keywords[source_dim] if kw in item)
        max_other_score = max(other_dim_scores.values()) if other_dim_scores else 0

        if max_other_score > source_score and max_other_score >= 2:
            removed_count += 1
            continue

        filtered_items.append(item)

    return filtered_items, removed_count


def _fix_editor_defects_format(summary: str) -> str:
    """
    后处理：去掉主编主要缺陷里的加粗格式，保持格式统一，同时限制最多6条
    有些审稿人的主要缺陷里有加粗标题（如**实验部分文字描述不足**：...），
    主编汇总时直接复制会导致有的条目加粗有的不加粗，格式不一致。
    同时主编经常列出超过6条主要缺陷，需要截断只保留前6条（最严重的）。
    """
    # 匹配"## 四、主要缺陷"到"## 五、"之间的内容
    # 注意：第一个分组只匹配"## 四、主要缺陷\n"，不要把第一条主要缺陷也分到prefix里，否则重新编号时第一条会被漏掉
    pattern = r'(## 四、主要缺陷\n)(.*?)(\n## 五、)'
    match = re.search(pattern, summary, re.DOTALL)
    if not match:
        return summary

    prefix = match.group(1)
    defects_content = match.group(2)
    suffix = match.group(3)

    # 按编号拆分成条目（匹配 "1. " "2. " 等）
    items = re.split(r'(?=\d+\.\s)', defects_content.strip())
    items = [item.strip() for item in items if item.strip()]

    original_count = len(items)
    modified = False

    # 先做维度归属校验：过滤掉明显属于其他维度的缺陷
    items, dimension_removed = _filter_editor_defects_by_dimension(items)
    if dimension_removed > 0:
        modified = True
        print(f"[后处理] 主编主要缺陷维度归属校验：去除了{dimension_removed}条越界内容")

    # 再去重：4个审稿人可能重复评价同一个问题
    items, dedup_removed = _deduplicate_items(items, threshold=0.4)
    if dedup_removed > 0:
        modified = True
        print(f"[后处理] 主编主要缺陷去重：去除了{dedup_removed}条重复内容")

    # 限制最多6条，超过的话只保留前6条（最严重的）
    if len(items) > 6:
        items = items[:6]
        modified = True
        print(f"[后处理] 主编主要缺陷数量修正：原来有{original_count}条，截断为6条")

    # 重新编号（总是执行，因为LLM生成的编号可能有重复或不连续）
    renumbered_items = []
    for i, item in enumerate(items, 1):
        item = re.sub(r'^\d+\.\s', f'{i}. ', item)
        renumbered_items.append(item)

    new_defects_content = '\n'.join(renumbered_items)

    # 去掉加粗格式：把**...**替换成...
    new_defects_content = re.sub(r'\*\*(.+?)\*\*', r'\1', new_defects_content)

    # 只要内容有变化（重新编号、截断、去加粗），就更新
    if new_defects_content == defects_content:
        return summary  # 内容完全一样，不需要修改

    new_summary = summary[:match.start()] + prefix + new_defects_content + suffix + summary[match.end():]
    if modified:
        print("[后处理] 主编主要缺陷格式修正：重新编号+去掉加粗+限制最多6条")
    else:
        print("[后处理] 主编主要缺陷格式修正：重新编号+去掉加粗")
    return new_summary


def _fix_editor_suggestions(summary: str) -> str:
    """
    后处理：过滤主编修改建议里的"代码"相关条目，并重新编号
    论文里看不到代码相关情况，不管什么代码都不该评价
    """
    # 匹配"## 五、修改建议清单"到"## 六、"之间的内容（去掉冲突说明后，修改建议是第五节）
    pattern = r'(## 五、修改建议清单.*?\n)(.*?)(\n## 六、)'
    match = re.search(pattern, summary, re.DOTALL)
    if not match:
        return summary

    prefix = match.group(1)
    suggestions_content = match.group(2)
    suffix = match.group(3)

    # 按编号拆分成条目（匹配 "1. " "2. " 等，允许编号前有空格）
    items = re.split(r'(?=\s*\d+\.\s)', suggestions_content.strip())
    items = [item.strip() for item in items if item.strip()]

    # 过滤掉包含"代码"的条目（论文里看不到代码相关情况，不管什么代码都不该评价）
    filtered_items = []
    removed_count = 0
    for item in items:
        if "代码" in item:
            removed_count += 1
            continue
        filtered_items.append(item)

    # 去重：4个审稿人可能重复提出相同的修改建议
    filtered_items, dedup_removed = _deduplicate_items(filtered_items, threshold=0.4)

    if removed_count == 0 and dedup_removed == 0:
        return summary  # 没有需要过滤或去重的条目

    # 重新编号：直接构造新的条目，不依赖正则替换（更可靠）
    renumbered_items = []
    for i, item in enumerate(filtered_items, 1):
        # 去掉原来的编号（允许前面有空格），然后加上新编号
        item_without_num = re.sub(r'^\s*\d+\.\s*', '', item)
        renumbered_items.append(f"{i}. {item_without_num}")

    new_suggestions_content = '\n'.join(renumbered_items)
    new_summary = summary[:match.start()] + prefix + new_suggestions_content + suffix + summary[match.end():]
    print(f"[后处理] 主编修改建议修正：删除了{removed_count}条'代码'相关条目，并重新编号")
    return new_summary


def _fix_editor_remove_conflict_section(summary: str) -> str:
    """
    后处理：去掉主编报告里的"审稿意见冲突说明"部分，并调整后面的编号
    四个审稿人负责的维度本来就不一样，每个维度只有一个审稿人，根本不存在"冲突"
    而且其他审稿人根本不评价创新性，不可能和创新性审稿人产生"创新性评价的冲突"
    这个部分完全是多余的，而且内容经常是编造的
    """
    # 匹配"## 五、审稿意见冲突说明"到"## 六、"之间的内容（包括标题行）
    pattern = r'## 五、审稿意见冲突说明.*?\n(.*?)(?=## 六、)'
    match = re.search(pattern, summary, re.DOTALL)
    if not match:
        return summary

    # 去掉"审稿意见冲突说明"部分
    summary = summary[:match.start()] + summary[match.end():]

    # 调整编号：六→五，七→六
    summary = summary.replace("## 六、修改建议清单", "## 五、修改建议清单")
    summary = summary.replace("## 七、最终结论", "## 六、最终结论")

    print("[后处理] 主编报告修正：去掉了'审稿意见冲突说明'部分，并调整了编号")
    return summary


def node_editor_summary(state: ReviewState) -> ReviewState:
    """主编汇总：汇总4份审稿意见，给出综合审稿报告"""
    # 直接用4个维度的初审意见
    innovation = state.get("innovation_review") or ""
    methodology = state.get("methodology_review") or ""
    experiment = state.get("experiment_review") or ""
    writing = state.get("writing_review") or ""

    llm = get_llm()
    prompt = get_editor_summary_prompt(state["topic"], innovation, methodology, experiment, writing)
    summary = llm.chat(prompt, system_prompt="你是一位资深的学术期刊领域主编（Area Chair），负责汇总多位审稿人的意见，给出最终的综合审稿报告和录用决定。", temperature=0.3)
    # 后处理：自动计算综合评分，替换掉LLM可能算错的加法
    summary = _fix_editor_total_score(summary)
    # 后处理：去掉主要缺陷里的加粗格式，保持格式统一，限制最多6条
    summary = _fix_editor_defects_format(summary)
    # 后处理：过滤修改建议里的"代码"相关条目
    summary = _fix_editor_suggestions(summary)
    # 后处理：去掉"审稿意见冲突说明"部分，并调整后面的编号
    summary = _fix_editor_remove_conflict_section(summary)
    # 后处理：把旧的维度名称"实验审稿人"替换成新的"论证与证据审稿人"（LLM可能用旧名称）
    summary = summary.replace("实验审稿人", "论证与证据审稿人")
    print("[汇总] 主编综合审稿报告完成")
    return {
        "editor_summary": summary,
        "phase": ReviewPhase.DONE,
        "full_transcript": [{
            "reviewer": "editor",
            "role": "主编综合审稿报告",
            "content": summary,
        }],
    }
