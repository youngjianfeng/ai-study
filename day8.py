import warnings
warnings.filterwarnings("ignore")

# =================  LangChain 负责 RAG 核心 =================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# =================  智谱原生 API 负责聊天 =================
from zhipuai import ZhipuAI

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FILE_PATH = "knowledge.txt"
# ==================================================

# 1. 加载本地文档
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

# 2. 文档切分（LangChain 功能）
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = splitter.split_documents(documents)

# 3. 向量化 + 存储（LangChain 功能）
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 4. 智谱官方客户端（最稳定）
client = ZhipuAI(api_key=API_KEY)

# ================== 聊天界面 ==================
print("=== 第8天：RAG 知识库机器人（纯 LangChain 版）===")

while True:
    question = input("你：")
    if question.lower() in ["退出", "exit"]:
        print("再见！")
        break

    # 检索相关知识（LangChain 最强功能）
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    # 构造提示词
    prompt = f"""
参考资料：
{context}

请根据上面的参考资料回答问题，不要瞎编。
问题：{question}
"""

    # 调用智谱大模型
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAI：", response.choices[0].message.content, "\n")


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from zhipuai import ZhipuAI

# # ================= 配置 =================
# API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
# FILE_PATH = "knowledge.txt"
# # ========================================

# # 加载文档
# loader = TextLoader(FILE_PATH, encoding="utf-8")
# documents = loader.load()

# # 切分文档
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# texts = splitter.split_documents(documents)
# docs = [t.page_content for t in texts]

# # 离线向量化（不联网、不下载、零等待）
# tfidf = TfidfVectorizer()
# tfidf_matrix = tfidf.fit_transform(docs)

# # 智谱
# client = ZhipuAI(api_key=API_KEY)

# print("=== RAG 离线知识库（国内完美运行）===")

# while True:
#     q = input("你：")
#     if q in ["退出", "exit"]:
#         break

#     # 离线匹配
#     q_vec = tfidf.transform([q])
#     sim = cosine_similarity(q_vec, tfidf_matrix)[0]
#     best = docs[sim.argmax()]

#     prompt = f"参考：{best}\n问题：{q}\n请根据参考回答："
#     res = client.chat.completions.create(model="glm-4-flash", messages=[{"role":"user","content":prompt}])
#     print("AI：", res.choices[0].message.content)