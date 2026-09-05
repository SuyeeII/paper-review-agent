# -*- coding: utf-8 -*-
"""
自动化审稿质量测试脚本
- 读取 tests/papers/ 下的论文
- 逐个调用 run_review() 跑完整审稿
- 检查输出质量（占位符残留、重复句、空缺陷、评分对齐、维度越界等）
- 输出质量报告
"""
import sys
import os
import re
import time
import json

# 插入项目根目录（tests的上一级）到sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from paper_review_agent.graph import run_review, format_review_result

# 复用 nodes.py 里的幻觉方法名列表，做独立断言（nodes过滤后是否有残留）
from paper_review_agent.nodes import HALLUCINATION_METHOD_KEYWORDS

# 幻觉断言只针对"具体方法名"（论文没引用而出现=编造）；
# "深度学习/机器学习"这类通用类别词是合理技术描述，不当作幻觉（避免误报）
GENERIC_METHOD_WORDS = {
    "CNN", "RNN", "Transformer", "BERT", "GPT", "SVM",
    "随机森林", "决策树", "贝叶斯", "马尔可夫", "蒙特卡洛",
    "强化学习", "深度学习", "机器学习",
}
SPECIFIC_METHOD_KEYWORDS = [m for m in HALLUCINATION_METHOD_KEYWORDS if m not in GENERIC_METHOD_WORDS]

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


def parse_paper(path: str) -> str:
    """解析论文文件（PDF/TXT/MD），提取文本内容"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if not HAS_PYPDF2:
            raise RuntimeError("PyPDF2 未安装，无法解析 PDF")
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    elif ext in (".txt", ".md", ".markdown"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise RuntimeError(f"不支持的文件格式：{ext}，仅支持 PDF / TXT / MD")

# ===== 质量检查规则 =====

# 1. 占位符残留（LLM照抄模板的痕迹）
PLACEHOLDER_PATTERNS = [
    r"（论文的哪一部分）",
    r"（具体是什么问题）",
    r"（具体问题）",
    r"（为什么这是缺陷）",
    r"（为什么是缺陷）",
    r"该维度的总体评分",
    r"（X/10）",
    r"（针对每个缺陷",
    r"（具体优点",
    r"（具体缺陷",
    r"\[论文中实际存在",
    r"\[论文中实际的",
]

# 2. 整句重复（同一句话连续出现两次）
REPEAT_PATTERN = r'(.{12,}?[。！？!?])\1'

# 3. 审稿意见结构完整性
REQUIRED_SECTIONS = ["**总体评价**", "**主要优点**", "**主要缺陷**", "**具体修改建议**", "**扣分说明**"]


def check_review_quality(review: str, dimension: str, results: list) -> None:
    """检查单个审稿人意见的质量"""
    if not review:
        results.append(f"  ❌ {dimension}审稿人：输出为空")
        return

    # 检查必备部分
    for section in REQUIRED_SECTIONS:
        if section not in review:
            results.append(f"  ❌ {dimension}审稿人：缺少{section}")

    # 检查占位符残留
    for pattern in PLACEHOLDER_PATTERNS:
        matches = re.findall(pattern, review)
        if matches:
            results.append(f"  ❌ {dimension}审稿人：占位符残留 {len(matches)}处 ({pattern[:30]})")

    # 检查整句重复
    for line in review.split('\n'):
        if re.search(REPEAT_PATTERN, line):
            results.append(f"  ❌ {dimension}审稿人：整句重复")

    # 检查空缺陷（主要缺陷为空）
    m = re.search(r'\*\*主要缺陷\*\*[：:]\s*\n(\s*)\*\*具体修改建议\*\*[：:]', review)
    if m:
        results.append(f"  ❌ {dimension}审稿人：主要缺陷为空")

    # 检查"未发现明显问题"与扣分说明矛盾（只查主要缺陷部分）
    m_defects = re.search(r'\*\*主要缺陷\*\*[：:]\s*\n(.*?)(\n\*\*[^\*]+\*\*[：:]|\Z)', review, re.DOTALL)
    m_deduction = re.search(r'\*\*扣分说明\*\*[：:]\s*(.*?)(\n(?=\*\*)|\Z)', review, re.DOTALL)
    if m_defects and m_deduction:
        if '未发现明显问题' in m_defects.group(1) and '未发现' not in m_deduction.group(1):
            results.append(f"  ❌ {dimension}审稿人：缺陷说'未发现问题'但扣分说明列了扣分理由")

    # 检查评分范围（1-10）
    m_score = re.search(r'\*\*总体评价\*\*[：:]\s*(\d+)/10', review)
    if m_score:
        score = int(m_score.group(1))
        if not (1 <= score <= 10):
            results.append(f"  ❌ {dimension}审稿人：评分超出范围 {score}/10")
    else:
        results.append(f"  ❌ {dimension}审稿人：总体评价缺少 X/10 分数")

    # 检查缺陷/建议条数一致性（缺陷有编号1.2.3.，建议也应该有）
    # 注意：当缺陷/建议为"（本维度未发现明显问题）"时无编号是正常的，跳过
    defects_start = review.find('**主要缺陷**')
    suggestions_start = review.find('**具体修改建议**')
    defects_section = review[defects_start:suggestions_start] if suggestions_start > defects_start >= 0 else ''
    suggestions_section = review[suggestions_start:] if suggestions_start >= 0 else ''
    n_defects = len(re.findall(r'^\d+\.\s', defects_section, re.M))
    n_suggestions = len(re.findall(r'^\d+\.\s', suggestions_section, re.M))
    if n_defects == 0 and '未发现明显问题' not in defects_section:
        results.append(f"  ❌ {dimension}审稿人：主要缺陷无编号条目")
    if n_suggestions == 0 and '未发现明显问题' not in suggestions_section:
        results.append(f"  ❌ {dimension}审稿人：具体修改建议无编号条目")

    # 检查写作审稿人维度越界（总体评价提到创新点）
    if dimension == '写作表达':
        m_overall = re.search(r'\*\*总体评价\*\*[：:]\s*(\d+)/10[，,]\s*(.*)', review)
        if m_overall:
            if re.search(r'创新点不够突出|创新性不足|新颖性不足', m_overall.group(2)):
                results.append(f"  ❌ {dimension}审稿人：总体评价出现创新维度措辞（维度越界）")


def _normalize(s: str) -> str:
    """规范化：去空格/连字符/下划线噪声，转小写（PDF提取常见）"""
    return re.sub(r'[\s\-_]+', '', s).lower()


def check_hallucination_residual(paper_text: str, full_report: str, results: list) -> None:
    """独立检查：最终报告里是否残留论文中不存在的具体方法名（幻觉）
    规范化匹配：PDF 提取文本可能有空格噪声（如 "Faster R -CNN"），需去噪后比较"""
    if not paper_text or not full_report:
        return
    paper_norm = _normalize(paper_text)
    report_norm = _normalize(full_report)
    for method in SPECIFIC_METHOD_KEYWORDS:
        method_norm = _normalize(method)
        if method_norm in paper_norm:
            continue  # 论文里真实存在（含提取噪声变体），不算幻觉
        if method_norm in report_norm:
            results.append(f"  ❌ 幻觉残留：报告中出现论文中不存在的方法名 '{method}'")


def check_editor_quality(state: dict, results: list) -> None:
    """检查主编综合报告质量"""
    summary = state.get("editor_summary") or ""
    if not summary:
        results.append("  ❌ 主编：输出为空")
        return

    for section in ["## 二、各维度评分", "## 三、主要优点", "## 四、主要缺陷", "## 五、修改建议清单", "## 六、最终结论"]:
        if section not in summary:
            results.append(f"  ❌ 主编：缺少{section}")

    # 检查占位符
    for pattern in PLACEHOLDER_PATTERNS:
        matches = re.findall(pattern, summary)
        if matches:
            results.append(f"  ❌ 主编：占位符残留 {len(matches)}处 ({pattern[:30]})")

    # 检查评分对齐：主编表格分数 = 初审分数
    review_scores = {}
    for dim, key in [("创新性", "innovation_review"), ("方法论", "methodology_review"),
                     ("论证与证据", "experiment_review"), ("写作表达", "writing_review")]:
        review = state.get(key) or ""
        m = re.search(r'\*\*总体评价\*\*[：:]\s*(\d+)/10', review)
        if m:
            review_scores[dim] = int(m.group(1))

    if len(review_scores) == 4:
        # 主编表格分数
        editor_scores = {}
        dim_map = {"创新性": r'\| 创新性 \|\s*(\d+)', "方法论": r'\| 方法论 \|\s*(\d+)',
                   "论证与证据": r'\| 论证与证据 \|\s*(\d+)', "写作表达": r'\| 写作表达 \|\s*(\d+)'}
        for dim, pattern in dim_map.items():
            m = re.search(pattern, summary)
            if m:
                editor_scores[dim] = int(m.group(1))

        if len(editor_scores) == 4:
            for dim in review_scores:
                if review_scores[dim] != editor_scores.get(dim):
                    results.append(f"  ❌ 主编评分与初审不一致：{dim} 初审{review_scores[dim]} vs 主编{editor_scores.get(dim)}")
                score = editor_scores.get(dim, 0)
                if not (1 <= score <= 10):
                    results.append(f"  ❌ 主编{dim}评分超出范围: {score}")
            # 总分校验
            total = sum(review_scores.values())
            m_total = re.search(r'\*\*综合评分\*\*\s*\|\s*\*\*(\d+)/40', summary)
            if m_total:
                if int(m_total.group(1)) != total:
                    results.append(f"  ❌ 主编综合评分错误：应为{total}，实际{m_total.group(1)}")
            else:
                results.append("  ❌ 主编：缺少综合评分（X/40）")
        else:
            results.append("  ❌ 主编：评分表格解析失败")
    else:
        results.append(f"  ⚠️ 初审评分提取不全（{len(review_scores)}/4），跳过主编评分对齐检查")

    # 检查修改建议清单格式（每条带【必须修改】/【建议修改】+ 编号）
    m_sugg_section = re.search(r'## 五、修改建议清单.*?(?=## 六、)', summary, re.DOTALL)
    if m_sugg_section:
        sugg_text = m_sugg_section.group(0)
        numbered_items = re.findall(r'^(\d+)\.\s+(\*\*【(必须|建议|可选)修改】\*\*|【(必须|建议|可选)修改】)', sugg_text, re.M)
        if not numbered_items:
            results.append("  ❌ 主编：修改建议清单缺少'【必须/建议修改】'编号条目")
        else:
            # 校验编号连续
            nums = [int(n) for n, *_ in numbered_items]
            if nums != list(range(1, len(nums) + 1)):
                results.append(f"  ❌ 主编：修改建议编号不连续 {nums}")
    else:
        results.append("  ❌ 主编：缺少修改建议清单")

    # 检查最终结论（格式：## 六、最终结论 标题 + 下一行 **小修**，或同行的 **最终结论**：小修）
    m_conclusion = re.search(r'##\s*六、最终结论\s*\n\s*\*{0,2}([^*\n]+?)\*{0,2}\s*\n', summary)
    if not m_conclusion:
        m_conclusion = re.search(r'\*\*最终结论\*\*[：:]\s*\*{0,2}([^*\n]+)', summary)
    if m_conclusion:
        conclusion = m_conclusion.group(1).strip()
        if not re.search(r'小修|大修|录用|拒稿|修改后录用|修改', conclusion):
            results.append(f"  ❌ 主编：最终结论异常：'{conclusion}'")
    else:
        results.append("  ❌ 主编：缺少最终结论")


def check_structure_quality(state: dict, results: list) -> None:
    """检查结构解析质量"""
    structure = state.get("paper_structure") or ""
    if not structure:
        results.append("  ❌ 结构解析：输出为空")
        return
    for section in ["## 论文基本信息", "### 1. 引言"]:
        if section not in structure:
            results.append(f"  ❌ 结构解析：缺少{section}")
    if '参考文献' not in structure:
        results.append("  ❌ 结构解析：缺少参考文献部分")
    # 检查占位符
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, structure):
            results.append(f"  ❌ 结构解析：占位符残留 ({pattern[:30]})")


def run_single_paper(paper_path: str) -> dict:
    """跑一篇论文，返回质量报告"""
    paper_text = parse_paper(paper_path)

    name = os.path.basename(paper_path)
    print(f"\n{'='*60}")
    print(f"📄 正在审稿: {name}")
    print(f"{'='*60}")

    # 跑完整审稿
    state = run_review(paper_text)
    results = []

    # 生成完整报告并保存（供人工复核）
    full_report = format_review_result(state)
    reports_dir = os.path.join(_SCRIPT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_name = os.path.splitext(name)[0] + ".md"
    with open(os.path.join(reports_dir, report_name), "w", encoding="utf-8") as f:
        f.write(full_report)

    # 检查各环节
    check_structure_quality(state, results)
    for dim, key in [("创新性", "innovation_review"), ("方法论", "methodology_review"),
                     ("论证与证据", "experiment_review"), ("写作表达", "writing_review")]:
        check_review_quality(state.get(key) or "", dim, results)
    check_editor_quality(state, results)
    # 幻觉方法名残留独立检查（覆盖4审稿人+主编全文）
    check_hallucination_residual(paper_text, full_report, results)

    # 输出结果
    if results:
        print(f"\n⚠️ {name} 发现 {len(results)} 个问题:")
        for r in results:
            print(r)
    else:
        print(f"\n✅ {name} 全部检查通过")

    return {
        "paper": name,
        "issues": results,
        "state": state,
    }


def main():
    papers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
    paper_files = sorted([
        os.path.join(papers_dir, f) for f in os.listdir(papers_dir)
        if f.endswith(('.txt', '.md', '.pdf'))
    ])

    if not paper_files:
        print("测试目录下没有论文文件")
        return

    print(f"共发现 {len(paper_files)} 篇测试论文")
    all_reports = []
    start = time.time()

    for paper in paper_files:
        report = run_single_paper(paper)
        all_reports.append(report)

    # 汇总
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"📊 汇总报告（耗时 {elapsed/60:.1f} 分钟）")
    print(f"{'='*60}")
    total_issues = 0
    for report in all_reports:
        n = len(report["issues"])
        total_issues += n
        status = "✅" if n == 0 else f"⚠️ {n}个问题"
        print(f"  {status} {report['paper']}")

    print(f"\n总问题数: {total_issues}")

    # 保存详细报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump([{"paper": r["paper"], "issues": r["issues"]} for r in all_reports],
                  f, ensure_ascii=False, indent=2)
    print(f"详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
