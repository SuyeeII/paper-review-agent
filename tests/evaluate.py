"""
系统质量评测脚本（对应评测体系 ③④ 层）

用法：
  1) 稳定性/自一致性评测（④层）：
     python tests/evaluate.py stability [论文文件名] [seed1 seed2 seed3 ...]
     默认：tests/papers/ 下第一篇论文，seed 42/2024/7 各跑一次
     输出：各维度评分、综合评分、最终结论的跨seed一致性

  2) 人工盲评模板生成（③层，黄金标准）：
     python tests/evaluate.py audit [论文文件名]
     从最新审稿报告提取所有"主要缺陷/具体修改建议"条目 → tests/audit_template.csv
     人工逐条对照原文标注：属实 / 夸大 / 编造
     填完后运行统计：
     python tests/evaluate.py audit-stats
     输出：意见准确率(Precision)、编造率 等
"""
import os
import re
import sys
import csv
import json
import time
import importlib

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from paper_review_agent.pdf_parser import parse_paper_file
from paper_review_agent.graph import run_review, format_review_result
import paper_review_agent.nodes as nodes_mod

DIMENSIONS = [
    ("创新性", "innovation_review"),
    ("方法论", "methodology_review"),
    ("论证与证据", "experiment_review"),
    ("写作表达", "writing_review"),
]


def _pick_paper(name: str = None) -> str:
    """选择论文：默认第一篇，或用名称模糊匹配"""
    papers_dir = os.path.join(_SCRIPT_DIR, "papers")
    files = sorted([os.path.join(papers_dir, f) for f in os.listdir(papers_dir)
                    if f.endswith(('.pdf', '.txt', '.md'))])
    if not files:
        raise RuntimeError("tests/papers/ 下没有论文")
    if not name:
        return files[0]
    for f in files:
        if name in os.path.basename(f):
            return f
    raise RuntimeError(f"未找到包含 '{name}' 的论文，可选: {[os.path.basename(f) for f in files]}")


def _extract_scores(state: dict) -> dict:
    """提取4个维度评分 + 综合评分 + 最终结论"""
    scores = {}
    for dim, key in DIMENSIONS:
        review = state.get(key) or ""
        m = re.search(r'\*\*总体评价\*\*[：:]\s*(\d+)/10', review)
        scores[dim] = int(m.group(1)) if m else None
    summary = state.get("editor_summary") or ""
    m_total = re.search(r'\*\*综合评分\*\*\s*\|\s*\*\*(\d+)/40', summary)
    scores["综合"] = int(m_total.group(1)) if m_total else None
    m_conc = re.search(r'##\s*六、最终结论\s*\n\s*\*{0,2}([^*\n]+?)\*{0,2}\s*\n', summary)
    if not m_conc:
        m_conc = re.search(r'\*\*最终结论\*\*[：:]\s*\*{0,2}([^*\n]+)', summary)
    scores["结论"] = m_conc.group(1).strip() if m_conc else None
    return scores


def evaluate_stability(paper_name: str, seeds: list) -> None:
    """稳定性/自一致性评测：同一篇论文不同seed重跑，统计输出波动"""
    paper_path = _pick_paper(paper_name)
    parsed = parse_paper_file(paper_path)
    paper_text = parsed["text"]
    print(f"📄 论文: {os.path.basename(paper_path)}（文本 {len(paper_text)} 字符）")
    print(f"🔁 重跑次数: {len(seeds)}（seed: {seeds}）\n")

    all_scores = []
    start = time.time()
    for seed in seeds:
        os.environ["PAPER_REVIEW_SEED"] = str(seed)
        importlib.reload(nodes_mod)  # 重新读取seed（模块级变量）
        state = run_review(
            paper_text,
            tables_md=parsed.get("tables_md", ""),
            images=parsed.get("images", 0),
            tables=parsed.get("tables", 0),
        )
        scores = _extract_scores(state)
        all_scores.append(scores)
        print(f"  seed={seed}: 创新{scores['创新性']} 方法{scores['方法论']} 论证{scores['论证与证据']} 写作{scores['写作表达']} | 综合{scores['综合']}/40 | 结论:{scores['结论']}")

    # 一致性统计
    print(f"\n{'='*60}")
    print("📊 稳定性评测结果")
    print(f"{'='*60}")

    dims = ["创新性", "方法论", "论证与证据", "写作表达", "综合"]
    for dim in dims:
        vals = [s[dim] for s in all_scores if s[dim] is not None]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            spread = max(vals) - min(vals)
            flag = "✅ 稳定" if spread <= 1 else ("⚠️ 波动" if spread <= 3 else "❌ 波动大")
            print(f"  {dim}: 数值{vals} | 均值{mean:.1f} | 方差{var:.2f} | 极差{spread} {flag}")

    conclusions = [s["结论"] for s in all_scores if s["结论"]]
    if conclusions:
        same = conclusions.count(conclusions[0]) == len(conclusions)
        print(f"  最终结论: {conclusions} | 一致性: {'✅ 全部一致' if same else '⚠️ 不一致'}")
        if not same:
            from collections import Counter
            print(f"    分布: {dict(Counter(conclusions))}")

    elapsed = time.time() - start
    print(f"\n⏱ 总耗时 {elapsed/60:.1f} 分钟（{len(seeds)} 次 × 6次LLM调用）")
    result = {"paper": os.path.basename(paper_path), "seeds": seeds, "runs": all_scores}
    with open(os.path.join(_SCRIPT_DIR, "stability_report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已保存: tests/stability_report.json")


def _extract_items(review: str, section: str) -> list:
    """提取某节的编号条目"""
    m = re.search(rf'\*\*{section}\*\*[：:]\s*\n(.*?)(?=\n\*\*[^*]+\*\*[：:]|\Z)', review, re.DOTALL)
    if not m:
        return []
    items = []
    for mm in re.finditer(r'^\d+\.\s+(.*)$', m.group(1), re.M):
        items.append(mm.group(1).strip())
    return items


def generate_audit_template(paper_name: str) -> None:
    """人工盲评模板：提取报告所有缺陷/建议条目 → CSV，供人工对照原文标注"""
    paper_path = _pick_paper(paper_name)
    parsed = parse_paper_file(paper_path)
    paper_text = parsed["text"]
    print(f"📄 论文: {os.path.basename(paper_path)}（正在审稿生成模板...）")
    state = run_review(paper_text)

    rows = []
    idx = 0
    for dim, key in DIMENSIONS:
        review = state.get(key) or ""
        for item in _extract_items(review, "主要缺陷"):
            idx += 1
            rows.append({"id": idx, "维度": dim, "类型": "缺陷", "内容": item,
                         "论文中可找到依据吗": "", "判定(属实/夸大/编造)": "", "备注": ""})
        for item in _extract_items(review, "具体修改建议"):
            idx += 1
            rows.append({"id": idx, "维度": dim, "类型": "建议", "内容": item,
                         "论文中可找到依据吗": "", "判定(属实/夸大/编造)": "", "备注": ""})

    out_path = os.path.join(_SCRIPT_DIR, "audit_template.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "维度", "类型", "内容", "论文中可找到依据吗", "判定(属实/夸大/编造)", "备注"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 已生成 {len(rows)} 条意见待人工标注: {out_path}")
    print("\n标注说明：")
    print("  对照论文原文逐条判断——")
    print("    属实：论文里确实有这个问题/建议合理可执行")
    print("    夸大：论文里有一点依据，但被放大/过度引申")
    print("    编造：论文里完全没有依据（幻觉）")
    print("  填完'判定'列后运行: python tests/evaluate.py audit-stats")


def audit_stats() -> None:
    """统计人工标注结果：Precision（意见准确率）、编造率"""
    csv_path = os.path.join(_SCRIPT_DIR, "audit_template.csv")
    if not os.path.exists(csv_path):
        print("未找到 tests/audit_template.csv，请先运行 generate-audit")
        return
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    judged = [r for r in rows if (r.get("判定(属实/夸大/编造)") or "").strip()]
    if not judged:
        print("还没有人工标注。请先打开 audit_template.csv 填写'判定'列。")
        return

    n = len(judged)
    cnt = {"属实": 0, "夸大": 0, "编造": 0}
    for r in judged:
        v = r["判定(属实/夸大/编造)"].strip()
        cnt[v] = cnt.get(v, 0) + 1

    # Precision = 属实/(全部) ，编造率 = 编造/全部
    precision = cnt["属实"] / n
    fabrication = cnt["编造"] / n

    print(f"\n{'='*60}")
    print("📊 人工盲评结果（黄金标准）")
    print(f"{'='*60}")
    print(f"已标注: {n} 条 | 属实 {cnt['属实']} | 夸大 {cnt['夸大']} | 编造 {cnt['编造']}")
    print(f"  意见准确率 Precision = {precision*100:.1f}%")
    print(f"  编造率 = {fabrication*100:.1f}%")
    print("\n解读：Precision 反映'系统说的问题里有多少是论文真实存在的'")
    print("      编造率是幻觉的直接量化，理想值为 0%")

    # 分维度统计
    print("\n分维度:")
    for dim in ["创新性", "方法论", "论证与证据", "写作表达"]:
        sub = [r for r in judged if r["维度"] == dim]
        if sub:
            ok = sum(1 for r in sub if r["判定(属实/夸大/编造)"].strip() == "属实")
            print(f"  {dim}: {ok}/{len(sub)} 属实 = {ok/len(sub)*100:.0f}%")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    mode = sys.argv[1]
    paper = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "stability":
        seeds = [int(x) for x in sys.argv[3:]] or [42, 2024, 7]
        evaluate_stability(paper, seeds)
    elif mode == "audit":
        generate_audit_template(paper)
    elif mode == "audit-stats":
        audit_stats()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
