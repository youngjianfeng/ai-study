# 第13天：AI Agent 全能版 → 思考 + 查资料 + 计算 + 查时间
import os
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FOLDER = "docs"
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
        print(f"✅ 已加载：{filename}")
    except:
        print(f"❌ 跳过：{filename}")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(all_docs)

# ===================== 2. 给 AI 装备工具 =====================
client = ZhipuAI(api_key=API_KEY)

def search_knowledge(question):
    """工具1：查知识库"""
    print("\n🔍 AI 使用工具：查资料")
    q_vec = tfidf.transform([question])
    return all_docs[cosine_similarity(q_vec, tfidf_matrix)[0].argmax()]

def calculate(expression):
    """工具2：计算数学题"""
    print("\n🧮 AI 使用工具：数学计算")
    try:
        return str(eval(expression))
    except:
        return "计算错误"

def get_time():
    """工具3：获取当前时间"""
    print("\n⏰ AI 使用工具：查时间")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================== 3. AI 大脑：判断用什么工具 =====================
def agent_think(user_input):
    prompt = f"""
用户问题：{user_input}
请判断需要使用哪个工具：
- 资料/知识/学习 → 回答：search
- 数学计算/加减乘除 → 回答：calc
- 时间/日期 → 回答：time
- 聊天 → 回答：none
只返回单词，不要加其他内容。
"""
    res = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content.strip()

# ===================== 4. 启动全能智能体 =====================
print("\n" + "="*60)
print("    第13天：AI Agent 全能版（多工具）")
print("="*60)

messages = [{"role": "system", "content": "你是专业全能AI助理。"}]

while True:
    user = input("\n你：")
    if user in ["退出", "exit"]:
        print("👋 再见！")
        break

    # AI 思考 → 选工具
    tool = agent_think(user)
    context = ""

    # 执行工具
    if tool == "search":
        context = search_knowledge(user)
    elif tool == "calc":
        context = calculate(user)
    elif tool == "time":
        context = get_time()

    # 最终回答
    prompt = f"工具结果：{context}\n用户问题：{user}"
    messages.append({"role": "user", "content": prompt})

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