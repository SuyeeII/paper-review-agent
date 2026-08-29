"""LLM 客户端封装
支持 DeepSeek / OpenAI 等兼容 OpenAI 接口的模型
"""
import os
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        # 优先用传入参数，其次用环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "未找到 API Key。请在 .env 文件中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，"
                "或者在初始化 LLMClient 时传入 api_key 参数。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, prompt: str, system_prompt: str = "你是一个专业的辩论赛选手。", temperature: float = None) -> str:
        """
        单次对话调用
        返回模型生成的文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def chat_with_history(self, messages: List[dict], temperature: float = None) -> str:
        """
        带历史对话的调用
        messages 格式: [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
