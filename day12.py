# 第12天：AI Agent 智能体（思考 + 决策 + 查RAG知识库）
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

# ===================== 1. 加载RAG知识库 =====================
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

# ===================== 2. AI Agent 核心功能 =====================
client = ZhipuAI(api_key=API_KEY)

def search_knowledge(question):
    """AI Agent 可以自己调用的【查资料】功能"""
    print("\n🔍 AI 思考：我需要查询知识库...")
    q_vec = tfidf.transform([question])
    best = all_docs[cosine_similarity(q_vec, tfidf_matrix)[0].argmax()]
    return best

# ===================== 3. AI Agent 主程序 =====================
print("\n" + "="*60)
print("    第12天：AI Agent 智能体（思考 + 查资料）")
print("="*60)

# 记忆
messages = [
    {"role": "system", "content": """
你是一个智能AI助手。
规则：
1. 如果问题需要知识、资料、学习内容 → 必须调用 search 工具查资料
2. 如果是日常聊天 → 直接回答
3. 查完资料再回答，不要瞎编
"""}
]

while True:
    user = input("\n你：")
    if user in ["退出", "exit"]:
        print("👋 再见！")
        break

    messages.append({"role": "user", "content": user})

    # ========== ✨ AI Agent 思考决策 ==========
    # 让AI判断是否需要查资料
    think_prompt = f"""
用户问题：{user}
请判断：是否需要查询知识库？
只回答：是 或 否
"""
    check = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role":"user","content":think_prompt}]
    )
    need_search = check.choices[0].message.content.strip()

    # 如果需要 → AI 自己调用查资料
    context = ""
    if "是" in need_search:
        context = search_knowledge(user)

    # ========== AI 生成最终回答 ==========
    prompt = f"参考资料：{context}\n用户问题：{user}"
    messages.append({"role": "user", "content": prompt})

    res = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=True
    )

    print("AI：", end="")
    ans = ""
    for chunk in res:
        c = chunk.choices[0].delta.content
        if c:
            print(c, end="")
            ans += c
    print()
    messages.append({"role": "assistant", "content": ans})