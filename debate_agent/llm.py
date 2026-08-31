"""LLM 客户端封装
支持 智谱GLM / OpenAI / DeepSeek 等兼容 OpenAI 接口的模型
V1.5 新增：超时控制 + 指数退避重试，解决 API 调用卡死问题
"""
import os
import time
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
        timeout: int = 120,
        max_retries: int = 3,
    ):
        # 优先用传入参数，其次用环境变量（智谱GLM优先，兼容旧DEEPSEEK变量名，最后OpenAI）
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("ZHIPU_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("ZHIPU_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or "glm-4-flash"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout          # 单次调用超时（秒）
        self.max_retries = max_retries  # 最大重试次数

        if not self.api_key:
            raise ValueError(
                "未找到 API Key。请在 .env 文件中设置 ZHIPU_API_KEY 或 OPENAI_API_KEY，"
                "或者在初始化 LLMClient 时传入 api_key 参数。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _call_with_retry(self, messages: List[dict], temperature: float = None) -> str:
        """
        带指数退避重试的 LLM 调用
        遇到网络错误、超时、限流时自动重试，最多 max_retries 次
        等待时间：1s → 2s → 4s（指数退避）
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"[LLM 重试] 第 {attempt+1} 次调用失败: {type(e).__name__}: {str(e)[:100]}，{wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[LLM 失败] 已重试 {self.max_retries} 次，最终失败: {type(e).__name__}: {str(e)[:200]}")
        # 所有重试都失败，抛出最后一次的错误
        raise last_error

    def chat(self, prompt: str, system_prompt: str = "你是一个专业的辩论赛选手。", temperature: float = None) -> str:
        """
        单次对话调用（带重试）
        返回模型生成的文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_with_retry(messages, temperature)

    def chat_with_history(self, messages: List[dict], temperature: float = None) -> str:
        """
        带历史对话的调用（带重试）
        messages 格式: [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        return self._call_with_retry(messages, temperature)


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
