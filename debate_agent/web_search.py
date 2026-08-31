"""V3 联网检索模块（Tavily）
Tavily 是专门为 AI Agent 设计的搜索 API，返回干净的正文摘要，适合 RAG 场景
免费额度：每月 1000 次搜索
注册地址：https://tavily.com
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class TavilySearch:
    """
    Tavily 联网搜索封装
    支持立场感知搜索：正方搜索支持正方的内容，反方搜索支持反方的内容
    """

    def __init__(self, api_key: str = None, max_results: int = 5):
        """
        Args:
            api_key: Tavily API Key，默认读环境变量 TAVILY_API_KEY
            max_results: 每次搜索返回的最大结果数
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self._client = None

        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                print("[Tavily] 未安装 tavily-python，请运行: pip install tavily-python")
            except Exception as e:
                print(f"[Tavily] 初始化失败: {e}")

    def is_available(self) -> bool:
        """检查 Tavily 是否可用"""
        return self._client is not None

    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """
        执行联网搜索
        Args:
            query: 搜索关键词
            max_results: 返回结果数，默认用初始化时的 max_results
        Returns:
            [{title, url, content, score}, ...]
        """
        if not self.is_available():
            return []

        try:
            k = max_results or self.max_results
            response = self._client.search(
                query=query,
                search_depth="basic",  # basic 更快，advanced 更全面但更慢
                max_results=k,
                include_answer=False,
            )
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                })
            return results
        except Exception as e:
            print(f"[Tavily] 搜索失败: {type(e).__name__}: {e}")
            return []

    def search_for_debate(
        self,
        topic: str,
        stance: str,
        stance_detail: str = None,
        opponent_argument: str = None,
        max_results: int = 5,
    ) -> List[Dict]:
        """
        为辩论场景做立场感知的联网搜索
        正方搜索支持正方的内容，反方搜索支持反方的内容

        Args:
            topic: 辩题
            stance: "affirmative" 正方 / "negative" 反方
            stance_detail: 立场表述（如"猫更适合当宠物"），用于构造搜索词
            opponent_argument: 对方上轮论点（攻辩时用，搜索反驳素材）
            max_results: 返回结果数

        Returns:
            [{title, url, content, score}, ...]
        """
        # 构造搜索查询：立场关键词 + 辩题
        if stance_detail:
            base_query = f"{stance_detail} 优点 好处 数据 研究"
        else:
            side_name = "正方" if stance == "affirmative" else "反方"
            base_query = f"{topic} {side_name} 论据 数据 研究"

        # 攻辩时加入对方论点关键词，搜索反驳素材
        if opponent_argument:
            # 取对方论点前100字作为关键词
            opponent_keywords = opponent_argument[:100].replace("\n", " ")
            query = f"{base_query} 反驳 {opponent_keywords}"
        else:
            query = base_query

        print(f"[Tavily] 搜索查询: {query[:80]}...")
        return self.search(query, max_results=max_results)


def format_search_results_for_prompt(results: List[Dict], stance: str) -> str:
    """
    把联网搜索结果格式化为 prompt 中的论据块
    """
    if not results:
        return ""

    side_name = "正方" if stance == "affirmative" else "反方"
    lines = [f"【联网检索到的可用论据（{side_name}立场，来自实时搜索）】"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")[:60]
        url = r.get("url", "")
        content = r.get("content", "")[:300]
        lines.append(f"{i}. 【{title}】（来源: {url}）")
        lines.append(f"   {content}")
    lines.append("你可以引用以上搜索结果中的数据和研究来支持你的论点，引用时请注明来源。")
    return "\n".join(lines)
