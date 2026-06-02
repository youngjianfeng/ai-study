# 语义分块 Semantic Chunking - Day1 核心代码
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import nltk

# 第一次运行自动下载分句模型
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 本地嵌入模型（和你RAG用的一样，不联网）
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
    similarity_threshold: float = 0.75
) -> list:
    """
    语义分块主函数
    :param text: 原始文本
    :param max_chunk_size: 最大块长度
    :param min_chunk_size: 最小块长度
    :param similarity_threshold: 相似度阈值（越低切越碎）
    :return: 语义切块列表
    """
    # 1. 分句
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # 2. 生成句向量
    sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])

    # 3. 按相似度滑动合并
    for i in range(1, len(sentences)):
        prev_emb = sentence_embeddings[i-1]
        curr_emb = sentence_embeddings[i]
        sim = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))

        # 相似度低 OR 超长 → 分块
        if sim < similarity_threshold or current_len + len(sentences[i]) > max_chunk_size:
            if current_len >= min_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

        current_chunk.append(sentences[i])
        current_len += len(sentences[i])

    # 加入最后一块
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks