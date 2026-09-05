"""
OpenReview 人工审稿对比评测（评测体系第⑤层：任务级/黄金对比）

流程：
  1) 从 OpenReview API 拉取有公开PDF+公开审稿意见的论文
  2) 下载论文PDF → 跑系统审稿（复用 run_review）
  3) 提取真人审稿意见的 weaknesses（真人缺陷意见）
  4) 用 LLM 做语义对齐：判断每条系统缺陷是否与任一真人意见"说的是同一件事"
  5) 计算指标：
     - Human Coverage (召回): 真人意见里被系统抓到的比例
     - Sys Precision: 系统缺陷里被真人意见支持的比例
     - F1: 综合重合度

用法：
  python tests/evaluate_openreview.py fetch [venue] [limit]   # 拉取并缓存论文列表（需网络）
  python tests/evaluate_openreview.py run [论文JSON序号]        # 对某篇论文跑完整对比评测
  python tests/evaluate_openreview.py run --all                # 对缓存的所有论文跑

数据缓存目录: tests/openreview_data/
"""
import os
import re
import sys
import json
import time
import urllib.request
import urllib.parse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from paper_review_agent.pdf_parser import parse_paper_file
from paper_review_agent.graph import run_review, format_review_result
from paper_review_agent.llm import get_llm

DATA_DIR = os.path.join(_SCRIPT_DIR, "openreview_data")
os.makedirs(DATA_DIR, exist_ok=True)

API_BASE = "https://api2.openreview.net"
FETCH_TIMEOUT = 30

DIMENSIONS = [
    ("创新性", "innovation_review"),
    ("方法论", "methodology_review"),
    ("论证与证据", "experiment_review"),
    ("写作表达", "writing_review"),
]


# ============ 1. 数据获取 ============

def http_get(url: str) -> dict:
    """GET JSON，带超时和UA"""
    req = urllib.request.Request(url, headers={"User-Agent": "paper-review-agent/1.0"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_venue_notes(venue: str, limit: int = 30) -> list:
    """
    拉取某会议有公开PDF的论文列表（note），只保留：
      - 有 title
      - 有 PDF 附件（content.pdf.value）
    """
    query = urllib.parse.quote(f'content.venue={venue}')
    url = f"{API_BASE}/notes?{query}&limit={limit}"
    print(f"[OpenReview] 拉取 {venue} 论文...")
    data = http_get(url)
    notes = data.get("notes", [])
    kept = []
    for n in notes:
        c = n.get("content", {})
        title = (c.get("title") or {}).get("value", "")
        pdf = (c.get("pdf") or {}).get("value", "")
        if title and pdf:
            kept.append({
                "id": n.get("id"),
                "title": title,
                "pdf": pdf,
                "venue": venue,
            })
    print(f"  共 {len(notes)} 篇，其中有PDF的 {len(kept)} 篇")
    return kept


def fetch_reviews(forum_id: str) -> list:
    """拉取某论文的全部 notes（含官方审稿 review）"""
    url = f"{API_BASE}/notes?forum={forum_id}"
    data = http_get(url)
    reviews = []
    for n in data.get("notes", []):
        c = n.get("content", {})
        # review note 特征：有 weaknesses 字段
        if "weaknesses" in c:
            def _v(k):
                v = c.get(k) or {}
                if isinstance(v, dict):
                    return v.get("value", "")
                return str(v)
            reviews.append({
                "id": n.get("id"),
                "strengths": _v("strengths"),
                "weaknesses": _v("weaknesses"),
                "rating": _v("rating"),
                "summary": _v("summary"),
            })
    return reviews


def cache_papers(venue: str, limit: int = 30) -> None:
    """拉取论文列表 + 每篇的真人审稿意见，缓存到本地"""
    papers = fetch_venue_notes(venue, limit)
    enriched = []
    for i, p in enumerate(papers):
        try:
            reviews = fetch_reviews(p["id"])
        except Exception as e:
            print(f"  [{i+1}/{len(papers)}] {p['title'][:40]}... 审稿意见拉取失败: {e}")
            reviews = []
        p["reviews"] = reviews
        p["_idx"] = i
        enriched.append(p)
        print(f"  [{i+1}/{len(papers)}] {p['title'][:40]}... 真人意见 {len(reviews)} 条")

    path = os.path.join(DATA_DIR, f"papers_{venue.replace(' ', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已缓存 {len(enriched)} 篇到 {path}")
    print("  下一步: python tests/evaluate_openreview.py run --all")


# ============ 2. 系统审稿 ============

def download_pdf(pdf_url: str, save_path: str) -> bool:
    """下载论文PDF"""
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "paper-review-agent/1.0"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        with open(save_path, "wb") as f:
            f.write(resp.read())
    return os.path.exists(save_path) and os.path.getsize(save_path) > 1000


def _extract_defects(review: str) -> list:
    """提取审稿人的主要缺陷条目"""
    m = re.search(r'\*\*主要缺陷\*\*[：:]\s*\n(.*?)(?=\n\*\*[^*]+\*\*[：:]|\Z)', review, re.DOTALL)
    if not m:
        return []
    items = [mm.group(1).strip() for mm in re.finditer(r'^\d+\.\s+(.*)$', m.group(1), re.M)]
    return [i for i in items if i and "未发现明显问题" not in i]


def run_system_review(pdf_path: str) -> dict:
    """系统审稿：返回系统缺陷条目列表 + 完整报告"""
    parsed = parse_paper_file(pdf_path)
    paper_text = parsed["text"]
    if not paper_text.strip():
        raise RuntimeError("PDF 无文本（可能是扫描件）")
    state = run_review(paper_text, tables_md=parsed.get("tables_md", ""),
                       images=parsed.get("images", 0), tables=parsed.get("tables", 0))
    sys_issues = []
    for dim, key in DIMENSIONS:
        for item in _extract_defects(state.get(key) or ""):
            sys_issues.append({"dim": dim, "text": item})
    return {"issues": sys_issues, "state": state, "text": paper_text}


# ============ 3. LLM 语义对齐 ============

def llm_align(human_weaknesses: list, sys_issues: list) -> list:
    """
    用 LLM 做语义对齐：系统缺陷 vs 真人意见
    返回匹配对列表 [(human_idx, sys_idx), ...]
    批处理：一次调用把所有条目给LLM，让它输出匹配对JSON
    """
    if not human_weaknesses or not sys_issues:
        return []
    llm = get_llm()
    human_list = "\n".join(f"[H{i}] {t}" for i, t in enumerate(human_weaknesses))
    sys_list = "\n".join(f"[S{i}] {t}" for i, t in enumerate(sys_issues))
    prompt = f"""你是学术论文审稿意见对齐专家。下面有两组审稿意见：
A组是真人审稿人提出的论文缺陷意见（H开头），B组是AI审稿系统提出的缺陷意见（S开头）。
请判断：B组中每条意见是否与A组中某条意见"说的是同一个问题"（语义等价，如"创新点不足"与"缺乏新颖性"算同一问题）。

【A组·真人审稿意见】
{human_list}

【B组·AI系统意见】
{sys_list}

请输出JSON格式的匹配对列表，格式如：
{{"matches": [{{"human": 0, "system": 1}}, ...]}}
匹配规则：
- 每条B组意见最多匹配一条A组意见；每条A组意见可被多条B组匹配
- 只输出语义等价的匹配，不要强行匹配
- 如果某条B组意见在A组中找不到等价问题，就不列入matches
- 只输出JSON，不要其他文字"""

    result = llm.chat(prompt, system_prompt="你是一个严谨的审稿意见对齐工具，只输出JSON。", temperature=0.1, max_tokens=1000, seed=42)
    try:
        # 提取JSON部分
        m = re.search(r'\{.*\}', result, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        pairs = []
        for match in data.get("matches", []):
            h, s = match.get("human"), match.get("system")
            if isinstance(h, int) and isinstance(s, int) and 0 <= h < len(human_weaknesses) and 0 <= s < len(sys_issues):
                pairs.append((h, s))
        return pairs
    except Exception as e:
        print(f"  [对齐失败] {e}: {result[:100]}")
        return []


# ============ 4. 指标计算与评测 ============

def evaluate_paper(p: dict, pdf_path: str) -> dict:
    """对单篇论文跑完整对比评测"""
    title = p["title"]
    print(f"\n{'='*60}")
    print(f"📄 {title[:60]}")
    print(f"{'='*60}")

    # 真人意见（取所有review的weaknesses，拆成条目）
    human_weaknesses = []
    for rv in p.get("reviews", []):
        w = rv.get("weaknesses", "")
        # 拆成条目（按编号/换行/分号）
        parts = re.split(r'\n+', w)
        for part in parts:
            part = re.sub(r'^\d+[\.、]\s*', '', part).strip()
            if len(part) >= 8:
                human_weaknesses.append(part)
    print(f"👤 真人意见: {len(p.get('reviews', []))} 条review，提取缺陷意见 {len(human_weaknesses)} 条")
    if not human_weaknesses:
        print("  ⚠️ 无真人缺陷意见，跳过")
        return None

    # 系统审稿
    print("🤖 系统审稿中（约2分钟）...")
    t0 = time.time()
    sys_result = run_system_review(pdf_path)
    sys_issues = sys_result["issues"]
    print(f"  系统缺陷意见 {len(sys_issues)} 条（耗时 {time.time()-t0:.0f}s）")
    if not sys_issues:
        print("  ⚠️ 系统无缺陷意见，跳过")
        return None

    # 语义对齐
    print("🔗 LLM语义对齐中...")
    pairs = llm_align(human_weaknesses, sys_issues)

    # 指标
    matched_human = set(h for h, _ in pairs)
    matched_sys = set(s for _, s in pairs)
    coverage = len(matched_human) / len(human_weaknesses)      # 真人意见召回
    precision = len(matched_sys) / len(sys_issues)             # 系统意见精确率
    f1 = 2 * coverage * precision / (coverage + precision) if (coverage + precision) > 0 else 0

    print(f"\n📊 对比结果:")
    print(f"  真人意见 {len(human_weaknesses)} 条 | 系统意见 {len(sys_issues)} 条 | 匹配 {len(pairs)} 对")
    print(f"  Human Coverage (真人意见召回率): {coverage*100:.0f}%  ({len(matched_human)}/{len(human_weaknesses)})")
    print(f"  Sys Precision (系统意见精确率):  {precision*100:.0f}%  ({len(matched_sys)}/{len(sys_issues)})")
    print(f"  F1 综合重合度: {f1*100:.0f}%")

    # 详细匹配展示
    if pairs:
        print("\n匹配详情:")
        for h, s in pairs:
            print(f"  [H{h}] {human_weaknesses[h][:50]}...")
            print(f"   ↕ [S{s}] {sys_issues[s]['text'][:50]}...")
    else:
        print("\n⚠️ 无任何匹配")

    return {
        "title": title,
        "human_count": len(human_weaknesses),
        "sys_count": len(sys_issues),
        "match_count": len(pairs),
        "coverage": coverage,
        "precision": precision,
        "f1": f1,
        "human": human_weaknesses,
        "sys": sys_issues,
        "pairs": [[h, s] for h, s in pairs],
    }


def run_all(venue: str = None) -> None:
    """对缓存的所有论文跑评测"""
    import glob
    files = sorted(glob.glob(os.path.join(DATA_DIR, "papers_*.json")))
    if not files:
        print("没有缓存数据，先运行: python tests/evaluate_openreview.py fetch \"ICLR 2025\" 20")
        return
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"\n缓存文件: {os.path.basename(fpath)}，共 {len(papers)} 篇")
        results = []
        for i, p in enumerate(papers):
            try:
                pdf_path = os.path.join(DATA_DIR, f"paper_{p['_idx']}.pdf")
                if not os.path.exists(pdf_path):
                    print(f"下载PDF: {p['title'][:40]}...")
                    download_pdf(p["pdf"], pdf_path)
                res = evaluate_paper(p, pdf_path)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"  [失败] {p['title'][:40]}...: {e}")
        # 汇总
        if results:
            print(f"\n{'='*60}")
            print("📈 OpenReview 对比评测汇总")
            print(f"{'='*60}")
            avg_cov = sum(r["coverage"] for r in results) / len(results)
            avg_prec = sum(r["precision"] for r in results) / len(results)
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            print(f"有效论文 {len(results)} 篇 | 真人意见共 {sum(r['human_count'] for r in results)} 条 | 系统意见共 {sum(r['sys_count'] for r in results)} 条")
            print(f"平均 Human Coverage: {avg_cov*100:.0f}%")
            print(f"平均 Sys Precision:  {avg_prec*100:.0f}%")
            print(f"平均 F1:             {avg_f1*100:.0f}%")
            report = {"venue": os.path.basename(fpath), "n": len(results),
                      "avg_coverage": avg_cov, "avg_precision": avg_prec, "avg_f1": avg_f1,
                      "papers": results}
            with open(os.path.join(DATA_DIR, "openreview_evaluation.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 已保存: tests/openreview_data/openreview_evaluation.json")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    mode = sys.argv[1]
    if mode == "fetch":
        venue = sys.argv[2] if len(sys.argv) > 2 else "ICLR 2025"
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        cache_papers(venue, limit)
    elif mode == "run":
        run_all()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
