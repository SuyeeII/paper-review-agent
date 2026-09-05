"""论文多视角审稿助手节点定义（多Agent并行评审架构）
4个维度审稿人并行评审 → 自我反思修正 → 主编汇总
"""
from typing import Optional
import re
from paper_review_agent.state import ReviewState, ReviewPhase
from paper_review_agent.llm import get_llm
from paper_review_agent.prompts import (
    get_paper_structure_prompt,
    get_innovation_review_prompt,
    get_methodology_review_prompt,
    get_experiment_review_prompt,
    get_writing_review_prompt,
    get_reflection_prompt,
    get_editor_summary_prompt,
    _clean_placeholder_brackets,
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


# ===== 后处理：检测并过滤幻觉的方法名称（防止LLM把示例里的方法套用到当前论文）=====

# 常见的、容易被LLM幻觉的方法/算法/模块名称列表
# 如果论文内容里没有这些名称，但审稿意见里出现了，就认为是幻觉，过滤掉包含这些名称的句子
HALLUCINATION_METHOD_KEYWORDS = [
    # 聚类/时序预测类（空调节能论文里的，容易被幻觉到其他论文）
    "K-means", "k-means", "K均值", "k均值", "LSTM", "lstm", "长短期记忆",
    # 优化算法类（PSO论文里的，容易被幻觉到其他论文）
    "IDPSO", "idpso", "PSO", "pso", "粒子群", "遗传算法", "GA", "模拟退火",
    # 目标检测类（YOLO论文里的，容易被幻觉到其他论文）
    "YOLO", "yolo", "C2f", "c2f", "C2f_CFE", "DLSCD", "MSDA", "CCFM",
    "Faster R-CNN", "Mask R-CNN", "DETR", "Swin Transformer",
    # 其他常见算法
    "CNN", "RNN", "Transformer", "BERT", "GPT", "SVM", "随机森林", "决策树",
    "贝叶斯", "马尔可夫", "蒙特卡洛", "强化学习", "深度学习", "机器学习",
]


def _filter_hallucinated_methods(review_content: str, paper_content: str, dimension: str) -> str:
    """
    检测并过滤审稿意见里幻觉的方法名称
    如果论文内容里没有某个方法名称，但审稿意见里出现了，就认为是幻觉
    过滤掉主要缺陷和修改建议里包含这些幻觉方法名称的条目
    """
    if not paper_content or not review_content:
        return review_content

    # 找出论文内容里实际存在的方法名称
    paper_methods = set()
    for method in HALLUCINATION_METHOD_KEYWORDS:
        if method in paper_content:
            paper_methods.add(method.lower())

    # 找出审稿意见里出现但论文里没有的方法名称（幻觉）
    hallucinated_methods = []
    for method in HALLUCINATION_METHOD_KEYWORDS:
        if method in review_content and method.lower() not in paper_methods:
            hallucinated_methods.append(method)

    if not hallucinated_methods:
        return review_content

    print(f"[幻觉检测] {dimension}审稿人：发现论文中不存在的方法名称 {hallucinated_methods}，正在过滤...")

    # 过滤主要缺陷和修改建议里包含幻觉方法名称的条目
    def _filter_section(section_text: str) -> tuple:
        items = re.split(r'(?=\d+\.\s)', section_text.strip())
        items = [item.strip() for item in items if item.strip()]
        filtered_items = []
        removed_count = 0
        for item in items:
            if any(method in item for method in hallucinated_methods):
                removed_count += 1
                continue
            filtered_items.append(item)
        # 重新编号
        renumbered = []
        for i, item in enumerate(filtered_items, 1):
            item = re.sub(r'^\d+\.\s', f'{i}. ', item)
            renumbered.append(item)
        return '\n'.join(renumbered), removed_count

    total_removed = 0

    # 过滤主要缺陷
    defect_match = re.search(r'(\*\*主要缺陷\*\*：?\n)(.*?)(?=\n\*\*具体修改建议\*\*|\n\*\*扣分说明\*\*|$)', review_content, re.DOTALL)
    if defect_match:
        prefix = defect_match.group(1)
        defect_text = defect_match.group(2)
        filtered_defects, removed = _filter_section(defect_text)
        total_removed += removed
        review_content = review_content[:defect_match.start(2)] + filtered_defects + review_content[defect_match.end(2):]

    # 过滤修改建议
    suggestion_match = re.search(r'(\*\*具体修改建议\*\*：?\n)(.*?)(?=\n\*\*扣分说明\*\*|$)', review_content, re.DOTALL)
    if suggestion_match:
        prefix = suggestion_match.group(1)
        suggestion_text = suggestion_match.group(2)
        filtered_suggestions, removed = _filter_section(suggestion_text)
        total_removed += removed
        review_content = review_content[:suggestion_match.start(2)] + filtered_suggestions + review_content[suggestion_match.end(2):]

    if total_removed > 0:
        print(f"[幻觉检测] {dimension}审稿人：过滤了 {total_removed} 条包含幻觉方法名称的内容")

    return review_content


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


def _deduplicate_repeated_sentences(review_text: str) -> str:
    """
    后处理：去掉LLM输出中整句重复的内容（同一句话连续出现两次）
    例如："论文第3节对XX分析不够深入，缺乏案例。论文第3节对XX分析不够深入，缺乏案例。"
    只删除完全相同的连续重复片段，不影响正常内容
    """
    if not review_text:
        return review_text

    # 通用去重：把连续出现的完全相同句子合并为一个
    def _dedup_segment(segment: str) -> str:
        # 按句子切分（中英文标点）
        sentences = re.split(r'(?<=[。！？.!?；;])', segment)
        result = []
        prev = ""
        for s in sentences:
            s_stripped = s.strip()
            if s_stripped and s_stripped == prev:
                continue  # 与上一句完全相同，跳过
            result.append(s)
            if s_stripped:
                prev = s_stripped
        return ''.join(result)

    # 对整段文本做处理：按行处理，但注意编号条目内重复句的处理
    lines = review_text.split('\n')
    new_lines = []
    for line in lines:
        # 如果一行内出现完全相同的子句重复（如"AAA。AAA。"），去重
        # 用循环反复去重，直到不再变化
        changed = True
        while changed:
            changed = False
            # 匹配 "X。X。" 或 "X！X！" 等完全重复模式（中间无其他字符）
            m = re.search(r'(.{8,}?[。！？!?])\1', line)
            if m:
                line = line[:m.start()] + m.group(1) + line[m.end():]
                changed = True
        new_lines.append(line)

    result = '\n'.join(new_lines)
    if result != review_text:
        print("[后处理] 审稿人输出存在整句重复，已去重")
    return result


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
    # 后处理过滤：检测并过滤幻觉的方法名称
    review = _filter_hallucinated_methods(review, state["topic"], "创新性")
    # 后处理过滤：清理输出格式占位符括号残留
    review = _clean_placeholder_brackets(review)
    # 后处理：去掉LLM输出的整句重复内容
    review = _deduplicate_repeated_sentences(review)
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
    # 后处理过滤：检测并过滤幻觉的方法名称
    review = _filter_hallucinated_methods(review, state["topic"], "方法论")
    # 后处理过滤：清理输出格式占位符括号残留
    review = _clean_placeholder_brackets(review)
    # 后处理：去掉LLM输出的整句重复内容
    review = _deduplicate_repeated_sentences(review)
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
    # 后处理过滤：检测并过滤幻觉的方法名称
    review = _filter_hallucinated_methods(review, state["topic"], "论证与证据")
    # 后处理过滤：清理输出格式占位符括号残留
    review = _clean_placeholder_brackets(review)
    # 后处理：去掉LLM输出的整句重复内容
    review = _deduplicate_repeated_sentences(review)
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
    # 后处理过滤：检测并过滤幻觉的方法名称
    review = _filter_hallucinated_methods(review, state["topic"], "写作表达")
    # 后处理过滤：清理输出格式占位符括号残留
    review = _clean_placeholder_brackets(review)
    # 后处理：去掉LLM输出的整句重复内容
    review = _deduplicate_repeated_sentences(review)
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

def _extract_reviewer_scores(innovation_review: str, methodology_review: str, experiment_review: str, writing_review: str) -> dict:
    """
    从4份初审意见中提取各维度评分（格式：**总体评价**：X/10，...）
    返回 {"innovation": int, "methodology": int, "experiment": int, "writing": int}
    提取失败的维度不返回（调用方校验数量）
    """
    scores = {}
    patterns = {
        "innovation": (innovation_review, r'\*\*总体评价\*\*\s*[:：]\s*(\d+)\s*/\s*10'),
        "methodology": (methodology_review, r'\*\*总体评价\*\*\s*[:：]\s*(\d+)\s*/\s*10'),
        "experiment": (experiment_review, r'\*\*总体评价\*\*\s*[:：]\s*(\d+)\s*/\s*10'),
        "writing": (writing_review, r'\*\*总体评价\*\*\s*[:：]\s*(\d+)\s*/\s*10'),
    }
    for dim, (text, pattern) in patterns.items():
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                scores[dim] = score
    return scores


def _fix_editor_scores_align(summary: str, reviewer_scores: dict) -> str:
    """
    后处理：主编报告"各维度评分"表格必须与初审审稿人评分一致。
    用初审分数覆盖主编表格里的分数（LLM可能擅自修改），并重算综合评分。
    """
    required = ["innovation", "methodology", "experiment", "writing"]
    if not all(dim in reviewer_scores for dim in required):
        return summary  # 任一维度提取失败就不强制覆盖

    # 各维度在表格中的中文名
    dim_names = {
        "innovation": "创新性",
        "methodology": "方法论",
        "experiment": "论证与证据",
        "writing": "写作表达",
    }

    # 1. 用初审分数覆盖表格分数
    changed = False
    for dim, name in dim_names.items():
        expected = reviewer_scores[dim]
        # 匹配 "| 创新性 | X |" 形式（只替换分数单元格，保留后面的说明列）
        pattern = r'(\|\s*' + re.escape(name) + r'\s*\|\s*)\d+(\s*\|)'
        match = re.search(pattern, summary)
        if match and int(match.group(0).split("|")[2].strip()) != expected:
            summary = summary[:match.start()] + match.group(1) + str(expected) + match.group(2) + summary[match.end():]
            changed = True

    # 2. 重算综合评分（= 四维之和）
    total = sum(reviewer_scores[dim] for dim in required)
    pattern_total = r'(\|\s*\*\*综合评分\*\*\s*\|\s*\*\*)\d+(/40\*\*\s*\|)'
    match_total = re.search(pattern_total, summary)
    if match_total:
        current_total = int(re.search(r'\*\*(\d+)/40\*\*', match_total.group(0)).group(1))
        if current_total != total:
            summary = summary[:match_total.start()] + match_total.group(1) + str(total) + match_total.group(2) + summary[match_total.end():]
            changed = True

    if changed:
        print(f"[后处理] 主编评分已对齐初审：{reviewer_scores['innovation']}+{reviewer_scores['methodology']}+{reviewer_scores['experiment']}+{reviewer_scores['writing']}={total}/40")
    return summary


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
        text = re.sub(r'[，。、；：！？""\'\'（）【】《》\s\.\,\;\:\!\?\(\)\[\]<>]', ' ', text)
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


def _fix_editor_advantages_format(summary: str) -> str:
    """
    后处理：对主编主要优点进行去重和重新编号
    4个审稿人可能重复评价同一个优点（如"结构完整""逻辑清晰"），需要去重
    """
    # 匹配"## 三、主要优点"到"## 四、"之间的内容
    pattern = r'(## 三、主要优点\n)(.*?)(\n## 四、)'
    match = re.search(pattern, summary, re.DOTALL)
    if not match:
        return summary

    prefix = match.group(1)
    advantages_content = match.group(2)
    suffix = match.group(3)

    # 按编号拆分成条目（匹配 "1. " "2. " 等）
    items = re.split(r'(?=\d+\.\s)', advantages_content.strip())
    items = [item.strip() for item in items if item.strip()]

    if len(items) <= 1:
        return summary  # 只有1条，不需要去重

    # 去重：4个审稿人可能重复评价同一个优点
    items, dedup_removed = _deduplicate_items(items, threshold=0.4)
    if dedup_removed > 0:
        print(f"[后处理] 主编主要优点去重：去除了{dedup_removed}条重复内容")

    # 重新编号（总是执行，因为去重后编号可能不连续）
    renumbered_items = []
    for i, item in enumerate(items, 1):
        item = re.sub(r'^\d+\.\s', f'{i}. ', item)
        renumbered_items.append(item)

    new_advantages_content = '\n'.join(renumbered_items)

    # 只要内容有变化（去重、重新编号），就更新
    if new_advantages_content == advantages_content:
        return summary  # 内容完全一样，不需要修改

    new_summary = summary[:match.start()] + prefix + new_advantages_content + suffix + summary[match.end():]
    print("[后处理] 主编主要优点格式修正：去重+重新编号")
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
    # 后处理：主编表格评分必须与初审审稿人评分一致（LLM可能擅自修改各维度分数）
    reviewer_scores = _extract_reviewer_scores(innovation, methodology, experiment, writing)
    summary = _fix_editor_scores_align(summary, reviewer_scores)
    # 后处理：对主要优点进行去重和重新编号
    summary = _fix_editor_advantages_format(summary)
    # 后处理：去掉主要缺陷里的加粗格式，保持格式统一，限制最多6条
    summary = _fix_editor_defects_format(summary)
    # 后处理：过滤修改建议里的"代码"相关条目
    summary = _fix_editor_suggestions(summary)
    # 后处理：去掉"审稿意见冲突说明"部分，并调整后面的编号
    summary = _fix_editor_remove_conflict_section(summary)
    # 后处理：把旧的维度名称"实验审稿人"替换成新的"论证与证据审稿人"（LLM可能用旧名称）
    summary = summary.replace("实验审稿人", "论证与证据审稿人")
    # 后处理：替换掉LLM可能直接保留的占位符（防止prompt示例里的占位符被当成内容）
    placeholder_patterns = [
        ("[论文中实际存在的方法/研究对象]", "现有相关方法"),
        ("[论文中实际存在的方法名称]", "本文提出的方法"),
        ("[论文中实际存在的方法]", "本文提出的方法"),
        ("[论文中实际存在的研究对象]", "本文研究对象"),
        ("[论文中实际存在的模块名称]", "本文提出的模块"),
        ("[论文中实际存在的实验名称]", "本文的实验"),
    ]
    placeholder_replaced = False
    for old, new in placeholder_patterns:
        if old in summary:
            summary = summary.replace(old, new)
            placeholder_replaced = True
    if placeholder_replaced:
        print("[后处理] 主编报告：替换了LLM未替换的占位符")
    # 后处理：清理输出格式占位符括号残留（如"（论文的哪一部分）"等）
    summary = _clean_placeholder_brackets(summary)
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
