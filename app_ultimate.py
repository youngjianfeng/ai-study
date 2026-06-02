# ========== 环境配置（必须在所有 import 之前） ==========
# 设置 HuggingFace 国内镜像，解决模型下载超时
# 注意：IDE 不会自动读取 ~/.zshrc，所以需要在代码里设置
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ===========================================================

# ==========================
# 第16天：终极全能AI助手
# 网页界面 + Chroma本地向量库 + 永久记忆 + Agent智能体
# ==========================
import json
import datetime
from flask import Flask, render_template, request, jsonify

# 1. Chroma 向量数据库（本地永久存储）
import chromadb
from chromadb.utils import embedding_functions

# 2. 文档读取与切分
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# 3. 智谱大模型
from zhipuai import ZhipuAI

# ===================== 【配置】 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
DOCS_FOLDER = "docs"
MEMORY_FILE = "memory.json"
CHROMA_DB_PATH = "./chroma_db"  # 向量库存在这里！
COLLECTION_NAME = "my_knowledge"
# =====================================================

app = Flask(__name__)
client = ZhipuAI(api_key=API_KEY)

# ===================== 1. 初始化 Chroma 向量库 =====================
# 本地永久向量库
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 离线向量模型（不联网、不下载、本地运行）
embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 获取/创建知识库
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding
)

# 文档切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# ===================== 2. 把文档导入向量库（只执行一次） =====================
def init_knowledge():
    if collection.count() > 0:
        print("✅ 向量库已存在，直接加载")
        return

    print("⏳ 首次启动，构建本地向量库...")
    all_documents = []

    for filename in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, filename)
        try:
            # 读取 TXT / PDF / DOCX
            if filename.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue

            # 切分文档
            splits = text_splitter.split_documents(loader.load())
            docs = [s.page_content for s in splits]
            all_documents.extend(docs)
            print(f"✅ 已导入：{filename}")

        except Exception as e:
            print(f"❌ 跳过：{filename}")

    # 存入向量库
    collection.add(
        documents=all_documents,
        ids=[f"id_{i}" for i in range(len(all_documents))]
    )
    print("✅ 向量库构建完成！")

# 初始化知识库
init_knowledge()

# ===================== 3. 工具函数 =====================
def search_rag(question):
    """RAG 检索：从本地向量库查资料"""
    res = collection.query(query_texts=[question], n_results=1)
    return res["documents"][0][0]

def calculate(expr):
    """数学计算"""
    try:
        return str(eval(expr))
    except:
        return "无法计算"

def get_time():
    """获取时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================== 4. 永久记忆 =====================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是专业全能AI助手，会查资料、会记忆、会使用工具。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# ===================== 5. AI Agent 思考 =====================
def agent_think(user_msg):
    prompt = f"""
用户问题：{user_msg}
请判断需要使用的工具：
- 需要知识/文档/资料 → search
- 需要数学计算 → calc
- 需要时间 → time
- 日常聊天 → none
只返回单词：search / calc / time / none
"""
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ===================== 6. 网页聊天界面 =====================
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🔥 第16天终极AI助手</title>
    <style>
        body{max-width:750px;margin:30px auto;font-family:Arial}
        .chat-box{height:550px;overflow-y:auto;border:1px solid #ddd;padding:20px;border-radius:10px;background:#fafafa}
        .msg{margin:10px 0;padding:12px 16px;border-radius:10px;max-width:75%}
        .user{background:#007bff;color:white;margin-left:auto}
        .bot{background:#e9e9eb;color:#222;margin-right:auto}
        .input-box{display:flex;margin-top:15px}
        input{flex:1;padding:14px;border-radius:8px;border:1px solid #ddd;font-size:15px}
        button{padding:14px 22px;background:#007bff;color:white;border:none;border-radius:8px;margin-left:8px;font-size:15px}
        .flex{display:flex}
    </style>
</head>
<body>
    <h2>🔥 第16天：终极全能AI助手（本地向量库版）</h2>
    <div class="chat-box" id="chat"></div>
    <div class="input-box">
        <input id="msg" placeholder="输入问题..." autocomplete="off">
        <button onclick="send()">发送</button>
    </div>

    <script>
        function addMsg(text, isUser){
            let div = document.createElement("div");
            div.className = "msg " + (isUser ? "user flex" : "bot");
            div.innerText = text;
            document.getElementById("chat").appendChild(div);
            document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
        }
        async function send(){
            let text = document.getElementById("msg").value.trim();
            if(!text) return;
            addMsg(text, true);
            document.getElementById("msg").value = "";
            
            let res = await fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({msg: text})
            });
            let data = await res.json();
            addMsg(data.reply, false);
        }
    </script>
</body>
</html>
'''

# ===================== 7. 聊天接口 =====================
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("msg")
    tool = agent_think(user_msg)
    context = ""

    # AI 自主调用工具
    if tool == "search":
        context = search_rag(user_msg)
    elif tool == "calc":
        context = calculate(user_msg)
    elif tool == "time":
        context = get_time()

    # 构造prompt
    prompt = f"参考资料：{context}\n用户问题：{user_msg}"
    messages.append({"role": "user", "content": prompt})

    # AI生成回答
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    save_memory(messages)

    return jsonify({"reply": reply})

# ===================== 启动 =====================
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)