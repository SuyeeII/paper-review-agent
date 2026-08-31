"""辩手 Agent
实现立论、攻辩、自我反思、驳论、总结五个阶段
V2 新增：RAG 论据检索，在发言前检索知识库中的真实论据注入 prompt
V3 新增：Tavily 联网检索，在发言前实时搜索最新论据注入 prompt
"""
from typing import List, Optional
from .llm import get_llm
from . import prompts
from .rag import KnowledgeBase, format_evidence_for_prompt
from .web_search import TavilySearch, format_search_results_for_prompt


class DebaterAgent:
    """辩手 Agent，代表正方或反方"""

    def __init__(self, side: str, stance_detail: str = None, llm=None):
        """
        Args:
            side: "affirmative" 正方 / "negative" 反方
            stance_detail: 该方立场的明确表述（如"猫更适合当宠物"），用于锚定立场
            llm: LLM 客户端实例
        """
        self.side = side
        self.side_name = "正方" if side == "affirmative" else "反方"
        self.stance_detail = stance_detail or ("支持本辩题" if side == "affirmative" else "反对本辩题")
        self.llm = llm or get_llm()
        # 记录自己的所有发言，用于反思和总结
        self.own_speeches: List[str] = []
        # V2 RAG 论据知识库（由外部设置，为 None 时不启用检索）
        self.evidence_kb: Optional[KnowledgeBase] = None
        # V3 Tavily 联网检索（由外部设置，为 None 时不启用联网搜索）
        self.web_search: Optional[TavilySearch] = None
        # 辩题（联网检索时用）
        self.topic: str = ""

    def retrieve_evidence(self, query: str, top_k: int = 3) -> str:
        """
        V2 RAG 论据检索
        从知识库中检索与 query 相关的论据，格式化为 prompt 可用的文本
        如果知识库不可用，返回空字符串
        """
        if self.evidence_kb is None or not self.evidence_kb.is_available():
            return ""
        evidences = self.evidence_kb.search(query, top_k=top_k)
        if not evidences:
            return ""
        return format_evidence_for_prompt(evidences, self.side_name)

    def retrieve_web_evidence(self, opponent_argument: str = None, max_results: int = 4) -> str:
        """
        V3 联网检索论据
        用 Tavily 实时搜索支持己方立场的论据，格式化为 prompt 可用的文本
        如果联网检索不可用，返回空字符串
        """
        if self.web_search is None or not self.web_search.is_available():
            return ""
        results = self.web_search.search_for_debate(
            topic=self.topic,
            stance=self.side,
            stance_detail=self.stance_detail,
            opponent_argument=opponent_argument,
            max_results=max_results,
        )
        if not results:
            return ""
        return format_search_results_for_prompt(results, self.side)

    def opening(self, topic: str) -> str:
        """立论陈词"""
        self.topic = topic  # V3: 保存辩题，联网检索时用
        prompt = prompts.get_opening_prompt(topic, self.side, self.stance_detail)
        # V2 RAG：检索论据注入 prompt
        evidence = self.retrieve_evidence(topic, top_k=3)
        # V3 联网检索：实时搜索论据注入 prompt
        web_evidence = self.retrieve_web_evidence(max_results=4)
        if evidence:
            prompt = prompt + "\n\n" + evidence
        if web_evidence:
            prompt = prompt + "\n\n" + web_evidence
        system = (
            f"你是一场辩论赛的{self.side_name}一辩，思维缜密，善于论证。"
            f"你的立场极其坚定，绝不妥协，绝不认同对方观点。"
            f"你的目标是说服评委接受你方立场，而不是追求客观中立。"
        )
        result = self.llm.chat(prompt, system_prompt=system)
        self.own_speeches.append(result)
        return result

    def clash(
        self,
        topic: str,
        round_num: int,
        opponent_previous: str,
        own_previous: str,
        reflection: Optional[str] = None,
    ) -> str:
        """
        攻辩发言
        参考对方上一轮发言和自己上一轮发言，进行反驳和延伸
        如果有 reflection（自我反思），则在发言中修正漏洞
        """
        prompt = prompts.get_clash_prompt(
            topic, self.side, self.stance_detail, round_num, opponent_previous, own_previous, reflection
        )
        # V2 RAG：检索论据注入 prompt（用辩题+对方上轮论点作为查询）
        evidence = self.retrieve_evidence(f"{topic} {opponent_previous[:200]}", top_k=3)
        # V3 联网检索：用对方论点做查询，搜索反驳素材
        web_evidence = self.retrieve_web_evidence(opponent_argument=opponent_previous, max_results=4)
        if evidence:
            prompt = prompt + "\n\n" + evidence
        if web_evidence:
            prompt = prompt + "\n\n" + web_evidence
        system = (
            f"你是辩论赛的{self.side_name}，反应敏捷，善于抓住对方漏洞进行反驳。"
            f"你的立场绝对坚定，绝不动摇。你的任务是摧毁对方论证，不是被对方说服。"
            f"即使对方说得有道理，也要从你的立场出发重新解读和反驳。"
        )
        result = self.llm.chat(prompt, system_prompt=system, temperature=0.8)
        self.own_speeches.append(result)
        return result

    def reflect(self, topic: str, own_speeches: List[str], opponent_speeches: List[str]) -> str:
        """
        自我反思（Self-Reflection 核心机制）
        批判自己之前的发言，找出漏洞，为下一轮修正做准备
        这个反思只有自己能看到，不会被对方知道

        Args:
            own_speeches: 自己到目前为止的所有发言（从 state 传入，不依赖实例状态）
            opponent_speeches: 对方到目前为止的所有发言
        """
        prompt = prompts.get_reflection_prompt(topic, self.side, self.stance_detail, own_speeches, opponent_speeches)
        system = (
            f"你是{self.side_name}的内部反思机制，要诚实地批判自己的论证，找出漏洞。"
            f"⚠️ 关键原则：反思的是论证方式和论据选择，绝不能否定自己的立场本身。"
            f"你的核心立场始终正确，需要改进的只是如何更好地论证它。反思结束后你将更加坚定。"
        )
        result = self.llm.chat(prompt, system_prompt=system, temperature=0.6)
        return result

    def rebuttal(self, topic: str, own_speeches: List[str], all_opponent_speeches: List[str], reflection: Optional[str] = None) -> str:
        """
        驳论陈词

        Args:
            own_speeches: 自己整场的所有发言（从 state 传入，用于锚定己方立场）
            all_opponent_speeches: 对方整场的所有发言
            reflection: 自我反思结果
        """
        prompt = prompts.get_rebuttal_prompt(topic, self.side, self.stance_detail, own_speeches, all_opponent_speeches, reflection)
        # V2 RAG：检索论据注入 prompt
        evidence = self.retrieve_evidence(topic, top_k=4)
        # V3 联网检索
        web_evidence = self.retrieve_web_evidence(max_results=4)
        if evidence:
            prompt = prompt + "\n\n" + evidence
        if web_evidence:
            prompt = prompt + "\n\n" + web_evidence
        system = (
            f"你是辩论赛的{self.side_name}，善于系统性地反驳对方论点。"
            f"你的立场坚如磐石，绝不动摇。你的目标是彻底摧毁对方的所有核心论点，"
            f"同时证明你方立场的绝对正确性。绝不妥协，绝不和稀泥。"
        )
        result = self.llm.chat(prompt, system_prompt=system)
        self.own_speeches.append(result)
        return result

    def closing(self, topic: str, own_speeches: List[str], opponent_all_speeches: List[str]) -> str:
        """
        总结陈词

        Args:
            own_speeches: 自己整场的所有发言（从 state 传入，不依赖实例状态）
            opponent_all_speeches: 对方整场的所有发言
        """
        prompt = prompts.get_closing_prompt(topic, self.side, self.stance_detail, own_speeches, opponent_all_speeches)
        system = (
            f"你是辩论赛的{self.side_name}四辩，善于总结升华，语言有感染力。"
            f"这是你最后一次说服评委的机会，立场必须绝对坚定。"
            f"你要证明你方赢得了这场辩论，对方彻底失败。"
            f"禁止任何中立或骑墙表述，最后一句话必须是坚定的立场声明。"
        )
        result = self.llm.chat(prompt, system_prompt=system, temperature=0.75)
        self.own_speeches.append(result)
        return result

    def summarize(self, topic: str, speeches_to_summarize: List[str], existing_summary: str = None) -> str:
        """
        V1.5 记忆摘要压缩
        把久远的辩论发言压缩成核心论点摘要，减少 token 消耗

        Args:
            topic: 辩题
            speeches_to_summarize: 需要被摘要的发言列表
            existing_summary: 已有的摘要（如果有，就整合更新）
        """
        prompt = prompts.get_summary_prompt(topic, self.side, self.stance_detail, speeches_to_summarize, existing_summary)
        system = (
            f"你是辩论赛的{self.side_name}记忆压缩助手，负责把历史发言压缩成核心论点摘要。"
            f"你的立场是：{self.stance_detail}。摘要必须保持立场坚定，不能出现中立或妥协表述。"
        )
        result = self.llm.chat(prompt, system_prompt=system, temperature=0.3)
        return result
