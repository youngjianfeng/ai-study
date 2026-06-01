# 第9天：PDF版RAG知识库（离线、国内完美运行）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # PDF专用
from zhipuai import ZhipuAI

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FILE_PATH = "test.pdf"  # 今天用PDF！
# ==================================================

# 1. 加载 PDF 文件（今天重点！）
loader = PyPDFLoader(FILE_PATH)
documents = loader.load()

# 2. 文档切分
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = splitter.split_documents(documents)
docs = [t.page_content for t in texts]

# 3. 离线向量化（零下载、不联网、国内秒启动）
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

# 4. 智谱AI
client = ZhipuAI(api_key=API_KEY)

print("="*50)
print("    第9天：PDF RAG 知识库（国内完美版）")
print("="*50)
print("输入 退出 结束\n")

# 开始问答
while True:
    question = input("你：")
    if question.lower() in ["退出", "exit"]:
        print("AI：再见！")
        break

    # 检索最相关的内容
    q_vector = tfidf.transform([question])
    similarities = cosine_similarity(q_vector, tfidf_matrix)[0]
    best_match = docs[similarities.argmax()]

    # 构造提示词
    prompt = f"""
参考资料：
{best_match}

问题：{question}
请根据参考资料简洁回答，不要瞎编。
"""

    # AI回答
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )

    print("AI：", response.choices[0].message.content, "\n")