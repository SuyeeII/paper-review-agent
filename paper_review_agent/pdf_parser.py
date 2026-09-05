"""
PDF/文本论文解析模块
- 文本提取：pdfplumber 优先（版式还原更好），PyPDF2 兜底
- 表格提取：pdfplumber extract_tables → Markdown 表格文本（注入审稿 prompt）
- 图表统计：统计图片/表格数量（供审稿人判断论文是否有实验图表支撑）
"""
import os

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


def _extract_text_with_pdfplumber(path: str) -> str:
    """pdfplumber 提取文本（含版式信息，还原度优于 PyPDF2）"""
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_text_with_pypdf2(path: str) -> str:
    """PyPDF2 提取文本（兜底）"""
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_tables_md(path: str, max_tables: int = 8, max_rows_per_table: int = 20) -> str:
    """
    pdfplumber 提取表格 → Markdown 表格
    表格过多/过长时截断，避免超出模型上下文
    碎片化表格（单列且行数<3，如PDF表格线识别错乱导致的文本碎片）不注入内容
    """
    if not HAS_PDFPLUMBER:
        return ""
    md_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            table_idx = 0
            for page in pdf.pages:
                for table in page.extract_tables():
                    if table_idx >= max_tables:
                        return "\n".join(md_parts)
                    rows = []
                    for row in table[:max_rows_per_table]:
                        cells = [(c or "").replace("\n", " ").strip() for c in row]
                        rows.append(cells)
                    # 过滤全空行
                    rows = [r for r in rows if any(c for c in r)]
                    if not rows:
                        continue
                    n_cols = max(len(r) for r in rows)
                    # 碎片表格判定：单列（或列数1）且行数<3 → 疑似表格线识别碎片，跳过内容只留数量
                    if n_cols <= 1 and len(rows) < 3:
                        table_idx += 1
                        continue
                    rows = [r + [""] * (n_cols - len(r)) for r in rows]
                    header = rows[0]
                    sep = ["---"] * n_cols
                    md_parts.append(f"### 表格{table_idx + 1}")
                    md_parts.append("| " + " | ".join(header) + " |")
                    md_parts.append("| " + " | ".join(sep) + " |")
                    for r in rows[1:]:
                        md_parts.append("| " + " | ".join(r) + " |")
                    md_parts.append("")
                    table_idx += 1
    except Exception:
        pass
    return "\n".join(md_parts)


def _count_figures_tables(path: str) -> dict:
    """统计图片/表格数量（pdfplumber 页面级）"""
    counts = {"images": 0, "tables": 0}
    if not HAS_PDFPLUMBER:
        return counts
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                counts["images"] += len(page.images or [])
                counts["tables"] += len(page.extract_tables() or [])
    except Exception:
        pass
    return counts


def parse_paper_file(path: str) -> dict:
    """
    统一入口：解析论文文件
    返回: {text, tables_md, images, tables, parser, warning}
    - text: 论文正文文本
    - tables_md: 表格转 Markdown（供审稿注入）
    - images/tables: 图/表数量
    - parser: 实际使用的解析器（pdfplumber/pypdf2）
    - warning: 非致命提示
    """
    ext = os.path.splitext(path)[1].lower()
    result = {"text": "", "tables_md": "", "images": 0, "tables": 0, "parser": "", "warning": ""}

    if ext in (".txt", ".md", ".markdown"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            result["text"] = f.read()
        result["parser"] = "text"
        return result

    if ext != ".pdf":
        result["warning"] = f"不支持的文件格式：{ext}，仅支持 PDF / TXT / MD"
        return result

    # PDF：pdfplumber 优先
    if HAS_PDFPLUMBER:
        try:
            result["text"] = _extract_text_with_pdfplumber(path)
            result["tables_md"] = _extract_tables_md(path)
            counts = _count_figures_tables(path)
            result["images"] = counts["images"]
            result["tables"] = counts["tables"]
            result["parser"] = "pdfplumber"
            if not result["text"].strip():
                result["warning"] = "pdfplumber 未提取到文本（可能是扫描件/图片型PDF），文本为空"
        except Exception as e:
            result["warning"] = f"pdfplumber 解析失败（{e}），回退 PyPDF2"

    if not result["text"] and HAS_PYPDF2:
        try:
            result["text"] = _extract_text_with_pypdf2(path)
            result["parser"] = "pypdf2"
        except Exception as e:
            result["warning"] = (result["warning"] + f"；PyPDF2 也失败：{e}").strip("；")

    if not result["text"] and not result["warning"]:
        result["warning"] = "未能提取到文本内容"

    return result
