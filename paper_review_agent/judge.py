"""评委 Agent
对整场辩论进行多维度评分，输出胜负和点评
"""
import json
import re
from typing import Dict, Any, Optional
from .llm import get_llm
from . import prompts


class JudgeAgent:
    """评委 Agent，独立于正反方，公正评分"""

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def judge(self, topic: str, affirmative_full: str, negative_full: str) -> Dict[str, Any]:
        """
        评分
        返回包含各维度分数、总分、胜者、点评的 dict
        """
        prompt = prompts.get_judge_prompt(topic, affirmative_full, negative_full)
        system = "你是一位经验丰富的辩论赛评委，评分公正客观，善于发现双方的亮点和不足。"

        result = self.llm.chat(prompt, system_prompt=system, temperature=0.3)

        # 解析 JSON 输出（模型可能输出 markdown 代码块包裹的 JSON）
        parsed = self._parse_json(result)
        return parsed

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        解析模型输出的 JSON
        处理可能的 markdown 代码块包裹、前后多余文本等情况
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 中的内容
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { 到最后一个 } 之间的内容
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # 都失败了，返回原始文本作为 comment
        return {
            "affirmative_scores": {},
            "negative_scores": {},
            "affirmative_total": 0,
            "negative_total": 0,
            "winner": "unknown",
            "margin": 0,
            "comment": text[:500],
            "parse_error": True,
            "raw_output": text,
        }
