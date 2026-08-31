"""多 Agent 辩论系统"""
from .graph import create_debate_graph, run_debate, format_debate_result
from .rag import KnowledgeBase

__all__ = ["create_debate_graph", "run_debate", "format_debate_result", "KnowledgeBase"]
