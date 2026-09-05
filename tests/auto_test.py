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

    # 检查写作审稿人维度越界（总体评价提到创新点）
    if dimension == '写作表达':
        m_overall = re.search(r'\*\*总体评价\*\*[：:]\s*(\d+)/10[，,]\s*(.*)', review)
        if m_overall:
            if re.search(r'创新点不够突出|创新性不足|新颖性不足', m_overall.group(2)):
                results.append(f"  ❌ {dimension}审稿人：总体评价出现创新维度措辞（维度越界）")


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
            # 总分校验
            total = sum(review_scores.values())
            m_total = re.search(r'\*\*综合评分\*\*\s*\|\s*\*\*(\d+)/40', summary)
            if m_total and int(m_total.group(1)) != total:
                results.append(f"  ❌ 主编综合评分错误：应为{total}，实际{m_total.group(1)}")
        else:
            results.append("  ❌ 主编：评分表格解析失败")
    else:
        results.append(f"  ⚠️ 初审评分提取不全（{len(review_scores)}/4），跳过主编评分对齐检查")


def check_structure_quality(state: dict, results: list) -> None:
    """检查结构解析质量"""
    structure = state.get("paper_structure") or ""
    if not structure:
        results.append("  ❌ 结构解析：输出为空")
        return
    for section in ["## 论文基本信息", "### 1. 引言", "### 6. 参考文献"]:
        if section not in structure:
            results.append(f"  ❌ 结构解析：缺少{section}")
    # 检查占位符
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, structure):
            results.append(f"  ❌ 结构解析：占位符残留 ({pattern[:30]})")


def run_single_paper(paper_path: str) -> dict:
    """跑一篇论文，返回质量报告"""
    with open(paper_path, encoding='utf-8') as f:
        paper_text = f.read()

    name = os.path.basename(paper_path)
    print(f"\n{'='*60}")
    print(f"📄 正在审稿: {name}")
    print(f"{'='*60}")

    # 跑完整审稿
    state = run_review(paper_text)
    results = []

    # 检查各环节
    check_structure_quality(state, results)
    for dim, key in [("创新性", "innovation_review"), ("方法论", "methodology_review"),
                     ("论证与证据", "experiment_review"), ("写作表达", "writing_review")]:
        check_review_quality(state.get(key) or "", dim, results)
    check_editor_quality(state, results)

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
        if f.endswith(('.txt', '.md'))
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
