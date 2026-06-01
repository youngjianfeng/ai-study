# 第15天：最终版 → 网页可视化 AI 助手（全能成品）
import os
import json
import datetime
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI

app = Flask(__name__)

# ===================== 配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
FOLDER = "docs"
MEMORY_FILE = "memory.json"
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

# ===================== 2. AI 工具 =====================
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

# ===================== 3. 记忆 =====================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是全能AI助手，拥有记忆、知识库、工具能力。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# ===================== 4. AI 思考 =====================
def agent_think(user_input):
    prompt = """
根据用户问题，判断需要使用的工具：
- 需要资料/知识 → search
- 需要计算 → calc
- 需要时间 → time
- 聊天 → none
只返回单词。
用户问题：{}
"""
    res = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt.format(user_input)}]
    )
    return res.choices[0].message.content.strip()

# ===================== 5. 网页界面 =====================
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>最终版 AI 助手</title>
    <style>
        body{max-width:700px;margin:20px auto;font-family:Arial}
        .chat{height:500px;overflow-y:auto;border:1px solid #ddd;padding:15px;border-radius:8px;background:#f9f9f9}
        .msg{margin:8px 0;padding:10px;border-radius:8px}
        .user{background:#007bff;color:white;text-align:right}
        .bot{background:#eee;color:black;text-align:left}
        .input{display:flex;margin-top:10px}
        input{flex:1;padding:12px;border-radius:6px;border:1px solid #ddd}
        button{padding:12px 20px;background:#007bff;color:white;border:none;border-radius:6px;margin-left:5px}
    </style>
</head>
<body>
    <h2>🔥 第15天最终版：全能AI助手</h2>
    <div class="chat" id="chat"></div>
    <div class="input">
        <input id="text" placeholder="输入消息...">
        <button onclick="send()">发送</button>
    </div>

    <script>
        function addMsg(text, isUser){
            let div = document.createElement("div");
            div.className = "msg " + (isUser ? "user" : "bot");
            div.innerText = text;
            document.getElementById("chat").appendChild(div);
            document.getElementById("chat").scrollTop = 1e9;
        }
        async function send(){
            let text = document.getElementById("text").value.trim();
            if(!text) return;
            addMsg(text, true);
            document.getElementById("text").value = "";
            let res = await fetch("/send", {
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body:JSON.stringify({msg:text})
            });
            let data = await res.json();
            addMsg(data.reply, false);
        }
    </script>
</body>
</html>
'''

@app.route('/send', methods=['POST'])
def send():
    user = request.json.get("msg")
    tool = agent_think(user)
    ctx = ""

    if tool == "search":
        ctx = search_knowledge(user)
    elif tool == "calc":
        ctx = calculate(user)
    elif tool == "time":
        ctx = get_time()

    prompt = f"工具结果：{ctx}\n问题：{user}"
    messages.append({"role": "user", "content": prompt})

    res = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )
    reply = res.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    save_memory(messages)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)