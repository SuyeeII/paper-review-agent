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
    query = urllib.parse.urlencode({"content.venue": venue})
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


def run_system_review(paper_text: str) -> dict:
    """系统审稿：返回系统缺陷条目列表 + 完整报告（直接吃论文全文文本）"""
    if not paper_text.strip():
        raise RuntimeError("论文文本为空")
    state = run_review(paper_text)
    sys_issues = []
    for dim, key in DIMENSIONS:
        for item in _extract_defects(state.get(key) or ""):
            sys_issues.append({"dim": dim, "text": item})
    return {"issues": sys_issues, "state": state, "text": paper_text}


# 系统审稿缓存：同一篇论文只审一次，重跑对齐时复用
_SYS_CACHE_PATH = os.path.join(DATA_DIR, "sys_reviews_cache.json")

def _load_sys_cache() -> dict:
    if os.path.exists(_SYS_CACHE_PATH):
        try:
            with open(_SYS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_sys_cache(cache: dict) -> None:
    with open(_SYS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


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

【A组·真人审稿意见】
{human_list}

【B组·AI系统意见】
{sys_list}

对B组中的【每一条】意见，判断它在A组中是否有"明确是同一个问题"的意见（语义等价，如"创新点不足"与"缺乏新颖性"算同一问题；但"基线选择太少"与"消融实验缺失"是不同问题）。
- 一条B组意见可以匹配0条或多条A组意见（当A组中一条长意见包含多个子问题时，可匹配多条）
- 只匹配"明确是同一个问题"的意见，语义相近但侧重不同的不算
- 如果某条B组意见在A组中找不到等价问题，给出空列表

请输出JSON格式，如：
{{"matches": [{{"system": 0, "human": [1, 3]}}, {{"system": 1, "human": []}}, ...]}}
每条B组意见都要有对应的条目。只输出JSON，不要其他文字"""

    result = llm.chat(prompt, system_prompt="你是一个严谨的审稿意见对齐工具，只输出JSON。", temperature=0.1, max_tokens=1500, seed=42)
    try:
        # 提取JSON部分
        m = re.search(r'\{.*\}', result, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        pairs = []
        for match in data.get("matches", []):
            s = match.get("system")
            h_list = match.get("human") or []
            if not isinstance(s, int) or not 0 <= s < len(sys_issues):
                continue
            for h in h_list:
                if isinstance(h, int) and 0 <= h < len(human_weaknesses):
                    pairs.append((h, s))
        return pairs
    except Exception as e:
        print(f"  [对齐失败] {e}: {result[:100]}")
        return []


# ============ 4. 指标计算与评测 ============

def evaluate_paper(p: dict, pdf_path: str = None, paper_text: str = None) -> dict:
    """对单篇论文跑完整对比评测（paper_text 直接给全文，否则解析 pdf_path）"""
    title = p["title"]
    print(f"\n{'='*60}")
    print(f"📄 {title[:60]}")
    print(f"{'='*60}")

    # 真人意见（取所有review的weaknesses，精细拆条：W1./编号/(1)/换行/分号）
    human_weaknesses = []
    for rv in p.get("reviews", []):
        w = rv.get("weaknesses") or ""
        parts = re.split(r'(?=\n\s*(?:W\d+[\.\:\)]|\(\d+\)|\d+[\.\)]|[-•]\s))|\n{2,}', w)
        for part in parts:
            part = re.sub(r'^(?:W\d+[\.\:\)]\s*|\(\d+\)\s*|\d+[\.\)]\s*|[-•]\s*)', '', part).strip()
            for sub in re.split(r';\s+', part):
                sub = sub.strip()
                if len(sub) >= 12:
                    human_weaknesses.append(sub)
    print(f"👤 真人意见: {len(p.get('reviews', []))} 条review，提取缺陷意见 {len(human_weaknesses)} 条")
    if not human_weaknesses:
        print("  ⚠️ 无真人缺陷意见，跳过")
        return None

    # 系统审稿（带缓存：同一篇只审一次）
    if paper_text is None and pdf_path:
        parsed = parse_paper_file(pdf_path)
        paper_text = parsed["text"]
    cache = _load_sys_cache()
    paper_key = p.get("paper_id") or title[:30]
    if paper_key in cache and cache[paper_key].get("issues"):
        sys_issues = cache[paper_key]["issues"]
        print(f"  系统意见 {len(sys_issues)} 条（命中缓存）")
    else:
        print("🤖 系统审稿中（约2分钟）...")
        t0 = time.time()
        sys_result = run_system_review(paper_text)
        sys_issues = sys_result["issues"]
        print(f"  系统缺陷意见 {len(sys_issues)} 条（耗时 {time.time()-t0:.0f}s）")
        cache[paper_key] = {"issues": sys_issues, "title": title}
        _save_sys_cache(cache)
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


def run_local(n: int = 3, text_dir: str = None) -> None:
    """
    本地数据模式：从 tests/openreview_data/ 读取
      papers.jsonl.gz + reviews.jsonl.gz + ICLR_2024.tar.gz（全文）
    评测：真人weaknesses vs 系统缺陷意见 的语义重合度
    """
    import gzip
    import glob as _glob
    import tarfile

    # 1. 读 reviews
    reviews_by_paper = {}
    for gz in _glob.glob(os.path.join(DATA_DIR, "reviews*.jsonl.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                reviews_by_paper.setdefault(r.get("paper_id"), []).append(r)
    print(f"📥 审稿意见: {sum(len(v) for v in reviews_by_paper.values())} 条，覆盖 {len(reviews_by_paper)} 篇论文")

    # 2. 读 papers
    papers = []
    for gz in _glob.glob(os.path.join(DATA_DIR, "papers*.jsonl.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    print(f"📄 论文: {len(papers)} 篇")

    # 3. 解压全文
    fulltexts = {}
    for tgz in _glob.glob(os.path.join(DATA_DIR, "ICLR_*.tar.gz")):
        print(f"📚 解压全文包: {os.path.basename(tgz)}")
        with tarfile.open(tgz, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".txt") and m.isfile():
                    try:
                        fobj = tar.extractfile(m)
                        fulltexts[os.path.splitext(os.path.basename(m.name))[0]] = fobj.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
    print(f"✅ 全文可用: {len(fulltexts)} 篇")

    # 4. 选评测论文：有全文 + 有≥2条review且weaknesses非空
    candidates = []
    for p in papers:
        pid = p.get("paper_id")
        if pid not in fulltexts or pid not in reviews_by_paper:
            continue
        revs = reviews_by_paper[pid]
        if len(revs) < 2:
            continue
        if not any((r.get("weaknesses") or "").strip() for r in revs):
            continue
        candidates.append(p)
    print(f"🎯 可评测论文: {len(candidates)} 篇")

    # 按真人评分分层随机抽样（评分分布：1=拒稿 ~ 10=强录用）
    import random
    random.seed(42)
    scored = []
    for p in candidates:
        revs = reviews_by_paper[p["paper_id"]]
        scores = [r.get("actual_score") for r in revs if isinstance(r.get("actual_score"), (int, float))]
        if scores:
            scored.append((sum(scores) / len(scores), p))
    if len(scored) > n:
        # 分层：低分(1-3)/中(4-7)/高(8-10) 各取 1/3
        buckets = {"low": [], "mid": [], "high": []}
        for avg_s, p in scored:
            if avg_s <= 3:
                buckets["low"].append((avg_s, p))
            elif avg_s <= 7:
                buckets["mid"].append((avg_s, p))
            else:
                buckets["high"].append((avg_s, p))
        picked = []
        per = max(1, n // 3)
        for b in ["low", "mid", "high"]:
            random.shuffle(buckets[b])
            picked.extend(buckets[b][:per])
        if len(picked) < n:
            rest = [x for x in scored if x not in picked]
            random.shuffle(rest)
            picked.extend(rest[: n - len(picked)])
    else:
        picked = scored
    picked.sort(key=lambda x: -x[0])
    print(f"🔀 分层抽样 {len(picked)} 篇（真人评分分布: " +
          " ".join(f"{s:.1f}" for s, _ in picked) + "）\n")

    results = []
    for i, (_, p) in enumerate(picked):
        pid = p["paper_id"]
        paper = {
            "paper_id": pid,
            "title": p.get("title", ""),
            "reviews": reviews_by_paper[pid],
            "abstract": p.get("abstract", ""),
        }
        print(f"[{i+1}/{len(picked)}] {paper['title'][:60]}...")
        try:
            res = evaluate_paper(paper, pdf_path=None, paper_text=fulltexts[pid])
            if res:
                results.append(res)
        except Exception as e:
            print(f"  [失败] {e}")
        # 每篇评测完立即保存中间结果，防止中断丢失
        _save_report(results)
    if results:
        print(f"\n{'='*60}")
        print("📈 OpenReview 对比评测汇总（系统 vs 真人审稿）")
        print(f"{'='*60}")
        avg_cov = sum(r["coverage"] for r in results) / len(results)
        avg_prec = sum(r["precision"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        print(f"有效论文 {len(results)} 篇 | 真人意见共 {sum(r['human_count'] for r in results)} 条 | 系统意见共 {sum(r['sys_count'] for r in results)} 条 | 匹配 {sum(r['match_count'] for r in results)} 对")
        print(f"平均 Human Coverage（真人意见召回率）: {avg_cov*100:.0f}%")
        print(f"平均 Sys Precision（系统意见精确率）:  {avg_prec*100:.0f}%")
        print(f"平均 F1 综合重合度: {avg_f1*100:.0f}%")
        _save_report(results)
        print(f"\n✅ 已保存: tests/openreview_data/openreview_evaluation.json")


def _save_report(results: list) -> None:
    """保存评测报告（每篇完成后调用，防止中断丢失）"""
    if not results:
        return
    avg_cov = sum(r["coverage"] for r in results) / len(results)
    avg_prec = sum(r["precision"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    report = {"n": len(results), "avg_coverage": avg_cov, "avg_precision": avg_prec, "avg_f1": avg_f1, "papers": results}
    with open(os.path.join(DATA_DIR, "openreview_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


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
    elif mode == "run-local":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_local(n)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
