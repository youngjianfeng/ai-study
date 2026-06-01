# 第11天：终极AI助理 = RAG知识库 + 长期记忆
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

# ===================== 1. 加载所有知识库 =====================
all_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

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
            continue
        
        splits = splitter.split_documents(loader.load())
        all_docs.extend([s.page_content for s in splits])
        print(f"✅ 已加载：{filename}")
    except:
        print(f"❌ 跳过：{filename}")

# 离线向量化
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(all_docs)

# ===================== 2. 初始化AI与记忆 =====================
client = ZhipuAI(api_key=API_KEY)

# 系统角色 + 长期记忆
messages = [
    {"role": "system", "content": "你是专业AI助理，会结合参考资料回答，简洁专业。"}
]

# ===================== 3. 启动对话 =====================
print("\n" + "="*60)
print("    第11天：终极智能AI助理（RAG + 记忆）")
print("="*60)

while True:
    user_input = input("你：")
    if user_input in ["退出", "exit"]:
        print("👋 再见！")
        break

    # 从知识库检索答案
    q_vec = tfidf.transform([user_input])
    best_doc = all_docs[cosine_similarity(q_vec, tfidf_matrix)[0].argmax()]

    # 构造带参考的提示词
    prompt = f"参考资料：\n{best_doc}\n\n用户问题：{user_input}"
    messages.append({"role": "user", "content": prompt})

    # AI 回答
    print("AI：", end="", flush=True)
    full_ans = ""
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=True
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_ans += content
    print()

    # 保存记忆
    messages.append({"role": "assistant", "content": full_ans})