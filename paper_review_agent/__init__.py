"""论文多视角审稿助手（多Agent并行评审架构）"""
from .graph import build_graph, run_review, format_review_result
# RAG / 联网检索模块暂未在审稿流程中启用，先注释避免引入额外依赖
# from .rag import KnowledgeBase
# from .web_search import TavilySearch

__all__ = ["build_graph", "run_review", "format_review_result"]
