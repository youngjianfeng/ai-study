# 第10天：终极版 → TXT + PDF + DOCX 多文件知识库
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FOLDER = "docs"
# ==================================================

all_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 自动读取文件夹里所有文件
for filename in os.listdir(FOLDER):
    path = os.path.join(FOLDER, filename)

    try:
        if filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue  # 其他格式跳过

        docs = loader.load()
        splits = splitter.split_documents(docs)
        all_docs.extend([s.page_content for s in splits])
        print(f"✅ 已加载：{filename}")

    except Exception as e:
        print(f"❌ 跳过 {filename}：{e}")

# 离线向量化
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(all_docs)

client = ZhipuAI(api_key=API_KEY)

print("\n=== 🚀 多格式知识库就绪（TXT+PDF+DOCX）===")

while True:
    q = input("你：")
    if q in ["退出", "exit"]:
        print("再见！")
        break

    q_vec = tfidf.transform([q])
    sim = cosine_similarity(q_vec, tfidf_matrix)[0]
    best = all_docs[sim.argmax()]

    prompt = f"""
参考资料：
{best}

问题：{q}
请根据资料简洁回答，不要瞎编。
"""

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    print("AI：", response.choices[0].message.content, "\n")