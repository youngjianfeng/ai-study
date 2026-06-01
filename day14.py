# 第14天：AI Agent + 永久记忆（自动保存/读取）
import os
import json
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FOLDER = "docs"
MEMORY_FILE = "memory.json"  # 记忆保存文件
# ==================================================

# ===================== 1. 加载知识库 =====================
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
    except:
        pass

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(all_docs)

# ===================== 2. 工具 =====================
client = ZhipuAI(api_key=API_KEY)

def search_knowledge(question):
    q_vec = tfidf.transform([question])
    return all_docs[cosine_similarity(q_vec, tfidf_matrix)[0].argmax()]

def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "计算错误"

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================== 3. 永久记忆功能 =====================
def load_memory():
    """读取历史记忆"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是全能AI助理，拥有永久记忆。"}]

def save_memory(messages):
    """保存对话记忆"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# ===================== 4. AI思考 =====================
def agent_think(user_input):
    prompt = f"""
用户问题：{user_input}
判断工具：
资料→search  计算→calc  时间→time  聊天→none
只返回单词。
"""
    r = client.chat.completions.create(model="glm-4-flash", messages=[{"role":"user","content":prompt}])
    return r.choices[0].message.content.strip()

# ===================== 5. 启动 =====================
print("="*60)
print("    第14天：AI Agent + 永久记忆")
print("="*60)

# 加载记忆
messages = load_memory()
print("✅ 已加载历史记忆！")

while True:
    user = input("\n你：")
    if user in ["退出", "exit", "关闭"]:
        save_memory(messages)
        print("💾 记忆已保存 → 再见！")
        break

    # 思考
    tool = agent_think(user)
    ctx = ""
    if tool == "search":
        print("\n🔍 查询知识库")
        ctx = search_knowledge(user)
    elif tool == "calc":
        print("\n🧮 计算中")
        ctx = calculate(user)
    elif tool == "time":
        print("\n⏰ 获取时间")
        ctx = get_time()

    prompt = f"工具结果：{ctx}\n问题：{user}"
    messages.append({"role": "user", "content": prompt})

    # 回答
    print("AI：", end="")
    ans = ""
    res = client.chat.completions.create(model="glm-4-flash", messages=messages, stream=True)
    for chunk in res:
        c = chunk.choices[0].delta.content
        if c:
            print(c, end="")
            ans += c
    print()

    messages.append({"role": "assistant", "content": ans})