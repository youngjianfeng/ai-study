# 进阶模块一：多路召回 + Rerank 增强RAG
import os
import json
import datetime
from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import jieba

# ===================== 全局配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
DOCS_FOLDER = "docs"
MEMORY_FILE = "memory.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_knowledge"

# 重排序模型（本地离线，无需联网）
RERANK_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RECALL = 5       # 多路召回总候选数
TOP_K_RERANK = 2       # 重排序后最终送入模型的数量

# ===================== 初始化基础组件 =====================
app = Flask(__name__)
zhipu_client = ZhipuAI(api_key=API_KEY)

# 1. 向量库 & 嵌入模型
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# 2. 文档切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# 3. 全局变量：纯文本块列表 + BM25实例
all_text_chunks = []
bm25 = None

# 4. 加载重排序模型
rerank_model = CrossEncoder(RERANK_MODEL_PATH)

# ===================== 1. 文档初始化 & 构建双检索索引 =====================
def init_document_and_index():
    """加载文档、构建向量库 + BM25关键词索引"""
    global all_text_chunks, bm25

    # 向量库已存在，直接读取文本块构建BM25
    if collection.count() > 0:
        print("✅ 向量库已存在，加载文本并构建BM25索引")
        all_data = collection.get(include=["documents"])
        all_text_chunks = all_data["documents"]
        # BM25需要分词列表（简单按空格/符号分词）
        # tokenized_corpus = [doc.split() for doc in all_text_chunks]
        tokenized_corpus = [list(jieba.cut(doc)) for doc in all_text_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return

    # 首次运行：加载文档、切分、入库
    print("⏳ 首次初始化文档与双检索索引...")
    for filename in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue

            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            chunk_texts = [c.page_content for c in chunks]
            all_text_chunks.extend(chunk_texts)
            print(f"✅ 加载文件：{filename}")
        except Exception as e:
            print(f"❌ 跳过 {filename}：{str(e)}")

    # 存入Chroma向量库
    ids = [f"chunk_{i}" for i in range(len(all_text_chunks))]
    collection.add(documents=all_text_chunks, ids=ids)

    # 构建BM25关键词检索索引
    tokenized_corpus = [doc.split() for doc in all_text_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ 向量库 + BM25 双索引构建完成")

# 执行初始化
init_document_and_index()

# ===================== 2. 多路召回 =====================
def multi_recall(query: str):
    """语义召回 + 关键词召回，合并候选结果"""
    candidates = set()

    # 1. 语义召回（Chroma向量检索）
    vec_res = collection.query(query_texts=[query], n_results=TOP_K_RECALL)
    for doc in vec_res["documents"][0]:
        candidates.add(doc)

    # 2. 关键词召回（BM25）
    # query_tokens = query.split()
    query_tokens = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(query_tokens)
    # 取分数最高的TOP_K候选
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:TOP_K_RECALL]
    for idx in top_bm25_idx:
        candidates.add(all_text_chunks[idx])

    return list(candidates)

# ===================== 3. 重排序 Rerank =====================
def rerank_candidates(query: str, candidates: list):
    """对召回结果重打分、排序，选出最优片段"""
    if not candidates:
        return ""
    # 构造 [问题, 候选文本] 对
    rank_pairs = [[query, cand] for cand in candidates]
    # 模型打分
    scores = rerank_model.predict(rank_pairs)
    # 按分数倒序排序
    sorted_pairs = sorted(zip(scores, candidates), reverse=True)
    # 取前N条拼接
    top_texts = [pair[1] for pair in sorted_pairs[:TOP_K_RERANK]]
    return "\n---\n".join(top_texts)

# ===================== 4. 工具函数、记忆、Agent思考（沿用原有逻辑） =====================
def calculate(expr):
    try:
        return str(eval(expr))
    except:
        return "无法计算"

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 记忆
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是全能AI助手，根据参考资料如实回答，不编造内容。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# Agent 工具判断
def agent_think(user_msg):
    prompt = f"""
用户问题：{user_msg}
判断使用工具：
知识资料查询 → search
数学计算 → calc
查询时间 → time
日常闲聊 → none
只返回单个单词：search / calc / time / none
"""
    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ===================== 5. 网页界面 & 接口 =====================
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>进阶RAG：多路召回+重排序</title>
    <style>
        body{max-width:750px;margin:30px auto;font-family:Arial}
        .chat-box{height:550px;overflow-y:auto;border:1px solid #ddd;padding:20px;border-radius:10px;background:#fafafa}
        .msg{margin:10px 0;padding:12px 16px;border-radius:10px;max-width:75%}
        .user{background:#007bff;color:white;margin-left:auto}
        .bot{background:#e9e9eb;color:#222;margin-right:auto}
        .input-box{display:flex;margin-top:15px}
        input{flex:1;padding:14px;border-radius:8px;border:1px solid #ddd;font-size:15px}
        button{padding:14px 22px;background:#007bff;color:white;border:none;border-radius:8px;margin-left:8px;font-size:15px}
    </style>
</head>
<body>
    <h2>🔥 进阶RAG | 多路召回 + Rerank 增强版</h2>
    <div class="chat-box" id="chat"></div>
    <div class="input-box">
        <input id="msg" placeholder="输入问题..." autocomplete="off">
        <button onclick="send()">发送</button>
    </div>

    <script>
        function addMsg(text, isUser){
            let div = document.createElement("div");
            div.className = "msg " + (isUser ? "user" : "bot");
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

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("msg")
    tool = agent_think(user_msg)
    context = ""

    if tool == "search":
        # 完整增强RAG链路：多路召回 → 重排序
        candidates = multi_recall(user_msg)
        context = rerank_candidates(user_msg, candidates)
    elif tool == "calc":
        context = calculate(user_msg)
    elif tool == "time":
        context = get_time()

    prompt = f"参考资料：{context}\n用户问题：{user_msg}，请根据资料简洁回答。"
    messages.append({"role": "user", "content": prompt})

    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    save_memory(messages)

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)