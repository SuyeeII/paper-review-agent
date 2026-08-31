"""V2 RAG 论据检索模块
支持：文档加载→切分→向量化→FAISS存储→检索
立场隔离：正方/反方论据库分开存储，检索时只检索己方立场的论据
"""
import os
import numpy as np
import faiss
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class KnowledgeBase:
    """
    论据知识库
    支持加载文档、切分、向量化、FAISS存储、检索
    每个知识库绑定一个立场（正方/反方），实现立场隔离
    """

    def __init__(
        self,
        stance: str,
        topic: str,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        """
        Args:
            stance: 立场标识，如"正方"或"反方"，用于立场隔离
            topic: 辩题，用于知识库命名
            api_key: embedding API key，默认读环境变量 EMBEDDING_API_KEY
            base_url: embedding API base_url，默认读环境变量 EMBEDDING_BASE_URL
            model: embedding 模型名，默认读环境变量 EMBEDDING_MODEL
        """
        self.stance = stance
        self.topic = topic
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        # 根据 base_url 自动选择默认 embedding 模型
        default_model = "text-embedding-3-small"
        if self.base_url and "bigmodel.cn" in self.base_url:
            default_model = "embedding-2"  # 智谱 embedding 模型
        self.model = model or os.getenv("EMBEDDING_MODEL") or default_model

        self.chunks: List[str] = []          # 切分后的文本块
        self.metadata: List[Dict] = []       # 每个块的元数据
        self.embeddings: Optional[np.ndarray] = None  # 向量矩阵
        self.index: Optional[faiss.IndexFlatL2] = None  # FAISS 索引

        self._client = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_available(self) -> bool:
        """检查 embedding API 是否可用"""
        return self._client is not None

    def load_documents(self, file_paths: List[str]) -> int:
        """
        加载文档文件（支持 .txt / .md）
        返回加载的文档数量
        """
        count = 0
        for path in file_paths:
            if not os.path.exists(path):
                print(f"[RAG] 文件不存在: {path}")
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 简单切分：按段落切，每段作为一个 chunk
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for i, para in enumerate(paragraphs):
                    # 如果段落太长，再按句子切
                    if len(para) > 500:
                        sentences = para.split('。')
                        current_chunk = ""
                        for sent in sentences:
                            if len(current_chunk) + len(sent) > 400:
                                if current_chunk.strip():
                                    self.chunks.append(current_chunk.strip())
                                    self.metadata.append({"source": os.path.basename(path), "stance": self.stance})
                                current_chunk = sent + "。"
                            else:
                                current_chunk += sent + "。"
                        if current_chunk.strip():
                            self.chunks.append(current_chunk.strip())
                            self.metadata.append({"source": os.path.basename(path), "stance": self.stance})
                    else:
                        self.chunks.append(para)
                        self.metadata.append({"source": os.path.basename(path), "stance": self.stance})
                count += 1
                print(f"[RAG] 加载文档: {path}，切分为 {len(paragraphs)} 段")
            except Exception as e:
                print(f"[RAG] 加载文件失败 {path}: {e}")
        return count

    def add_text(self, text: str, source: str = "manual"):
        """手动添加一段文本作为论据"""
        self.chunks.append(text)
        self.metadata.append({"source": source, "stance": self.stance})

    def build_index(self) -> bool:
        """
        对所有 chunk 做向量化，构建 FAISS 索引
        返回是否成功
        """
        if not self.is_available():
            print("[RAG] 未配置 embedding API，无法构建索引。请在 .env 中设置 EMBEDDING_API_KEY")
            return False

        if not self.chunks:
            print("[RAG] 没有可索引的文本块")
            return False

        print(f"[RAG] 正在对 {len(self.chunks)} 个文本块做向量化（模型: {self.model}）...")

        try:
            # 批量向量化（OpenAI 兼容接口一次最多处理多个输入）
            response = self._client.embeddings.create(
                model=self.model,
                input=self.chunks,
            )
            self.embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)

            # 构建 FAISS L2 距离索引
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)

            print(f"[RAG] 索引构建成功，维度: {dim}，向量数: {self.index.ntotal}")
            return True
        except Exception as e:
            print(f"[RAG] 向量化失败: {type(e).__name__}: {e}")
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索与 query 最相关的 top_k 个论据
        返回 [{content, source, score}, ...]
        """
        if self.index is None or self.embeddings is None:
            return []

        try:
            # 查询向量化
            response = self._client.embeddings.create(
                model=self.model,
                input=[query],
            )
            query_vec = np.array([response.data[0].embedding], dtype=np.float32)

            # FAISS 检索
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vec, k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                results.append({
                    "content": self.chunks[idx],
                    "source": self.metadata[idx].get("source", "unknown"),
                    "score": float(distances[0][i]),  # L2 距离，越小越相似
                })
            return results
        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return []


def format_evidence_for_prompt(evidences: List[Dict], stance: str) -> str:
    """
    把检索到的论据格式化为 prompt 中的一部分
    """
    if not evidences:
        return ""

    lines = [f"【可引用的论据（{stance}立场，来自知识库检索）】"]
    for i, ev in enumerate(evidences, 1):
        lines.append(f"{i}. （来源: {ev['source']}）{ev['content']}")
    lines.append("你可以引用以上论据来支持你的论点，引用时请注明数据来源。")
    return "\n".join(lines)
