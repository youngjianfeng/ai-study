# 进阶模块二：分层分块 + 层级RAG（结合多路召回+重排序）
import os
import json
import datetime
import jieba
from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from zhipuai import ZhipuAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ===================== 全局配置 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
DOCS_FOLDER = "docs"
MEMORY_FILE = "memory.json"
CHROMA_DB_PATH = "./chroma_hier_db"
COLLECTION_COARSE = "coarse_chunk"   # 粗块集合
COLLECTION_FINE = "fine_chunk"      # 细块集合

# 分块参数
COARSE_CHUNK_SIZE = 800
COARSE_OVERLAP = 100
FINE_CHUNK_SIZE = 300
FINE_OVERLAP = 50

# 检索&重排序参数
TOP_K_COARSE = 3        # 粗块召回数量
TOP_K_RECALL = 8        # 细块多路召回数量
TOP_K_RERANK = 3        # 最终精选数量
RERANK_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ===================== 初始化基础组件 =====================
app = Flask(__name__)
zhipu_client = ZhipuAI(api_key=API_KEY)

# 向量库 & 嵌入模型
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 获取两级向量集合
coarse_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_COARSE,
    embedding_function=embedding_func
)
fine_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_FINE,
    embedding_function=embedding_func
)

# 两级分块器
coarse_splitter = RecursiveCharacterTextSplitter(
    chunk_size=COARSE_CHUNK_SIZE,
    chunk_overlap=COARSE_OVERLAP
)
fine_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FINE_CHUNK_SIZE,
    chunk_overlap=FINE_OVERLAP
)

# 全局存储：细块文本、BM25索引、粗细块映射关系
all_fine_chunks = []
bm25_fine = None
# key:粗块ID  value:对应所有细块下标列表
coarse2fine_map = {}

# 加载重排序模型
rerank_model = CrossEncoder(RERANK_MODEL_PATH)

# ===================== 1. 文档初始化 & 构建分层索引 =====================
def init_hierarchical_index():
    global all_fine_chunks, bm25_fine, coarse2fine_map

    # 索引已存在，直接加载映射与文本
    if coarse_collection.count() > 0 and fine_collection.count() > 0:
        print("✅ 分层索引已存在，直接加载")
        # 读取所有细块
        fine_data = fine_collection.get(include=["documents", "metadatas"])
        all_fine_chunks = fine_data["documents"]
        # 恢复粗细块映射
        for idx, meta in enumerate(fine_data["metadatas"]):
            coarse_id = meta.get("coarse_id")
            if coarse_id not in coarse2fine_map:
                coarse2fine_map[coarse_id] = []
            coarse2fine_map[coarse_id].append(idx)
        # 构建BM25
        token_corpus = [list(jieba.cut(doc)) for doc in all_fine_chunks]
        bm25_fine = BM25Okapi(token_corpus)
        return

    print("⏳ 首次构建分层分块索引...")
    coarse_id_counter = 0
    fine_id_counter = 0

    # 遍历文档
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
            # 第一步：生成粗块
            coarse_chunks = coarse_splitter.split_documents(docs)
            print(f"📄 {filename} 生成 {len(coarse_chunks)} 个粗块")

            for coarse_chunk in coarse_chunks:
                c_id = f"coarse_{coarse_id_counter}"
                coarse_id_counter += 1
                c_text = coarse_chunk.page_content

                # 粗块入库
                coarse_collection.add(
                    documents=[c_text],
                    ids=[c_id]
                )

                # 第二步：粗块再切分为细块
                fine_sub_chunks = fine_splitter.split_text(c_text)
                for f_text in fine_sub_chunks:
                    f_id = f"fine_{fine_id_counter}"
                    fine_id_counter += 1
                    all_fine_chunks.append(f_text)

                    # 细块入库，元数据绑定所属粗块ID
                    fine_collection.add(
                        documents=[f_text],
                        ids=[f_id],
                        metadatas=[{"coarse_id": c_id}]
                    )
                    # 维护映射关系
                    if c_id not in coarse2fine_map:
                        coarse2fine_map[c_id] = []
                    coarse2fine_map[c_id].append(len(all_fine_chunks)-1)

        except Exception as e:
            print(f"❌ 跳过 {filename}：{str(e)}")

    # 构建细块BM25索引
    token_corpus = [list(jieba.cut(doc)) for doc in all_fine_chunks]
    bm25_fine = BM25Okapi(token_corpus)
    print("✅ 分层向量库 + BM25 索引构建完成")

# 执行初始化
init_hierarchical_index()

# ===================== 2. 层级检索：粗块定位 → 细块召回 =====================
def hierarchical_retrieval(query: str):
    # 1. 第一层：检索粗块，定位范围
    coarse_res = coarse_collection.query(
        query_texts=[query],
        n_results=TOP_K_COARSE
    )
    hit_coarse_ids = set()
    for meta in coarse_res["metadatas"][0]:
        if meta is not None and "id" in meta:
            hit_coarse_ids.add(meta["id"])

    # 2. 根据命中粗块，收集对应细块下标
    candidate_fine_idx = set()
    for cid in hit_coarse_ids:
        if cid in coarse2fine_map:
            for idx in coarse2fine_map[cid]:
                candidate_fine_idx.add(idx)
    if not candidate_fine_idx:
        return []

    # 3. 第二层：在范围内做多路召回（语义+BM25）
    candidates = set()
    # 语义召回
    fine_res = fine_collection.query(
        query_texts=[query],
        n_results=TOP_K_RECALL
    )
    for doc in fine_res["documents"][0]:
        candidates.add(doc)

    # BM25关键词召回（中文分词）
    query_tokens = list(jieba.cut(query))
    bm25_scores = bm25_fine.get_scores(query_tokens)
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda x: bm25_scores[x],
        reverse=True
    )[:TOP_K_RECALL]
    for idx in top_bm25_idx:
        if idx in candidate_fine_idx:
            candidates.add(all_fine_chunks[idx])

    return list(candidates)

# ===================== 3. 重排序 =====================
def rerank_candidates(query: str, candidates: list):
    if not candidates:
        return ""
    rank_pairs = [[query, cand] for cand in candidates]
    scores = rerank_model.predict(rank_pairs)
    sorted_items = sorted(zip(scores, candidates), reverse=True)
    top_texts = [item[1] for item in sorted_items[:TOP_K_RERANK]]
    return "\n---\n".join(top_texts)

# ===================== 4. 工具、记忆、Agent 判断逻辑 =====================
def calculate(expr):
    try:
        return str(eval(expr))
    except:
        return "无法计算"

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 对话记忆
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是全能AI助手，严格根据参考资料回答，不编造内容。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# Agent 工具决策
def agent_think(user_msg):
    prompt = f"""
用户问题：{user_msg}
判断工具类型：
查询知识/文档 → search
数学计算 → calc
查询时间 → time
日常闲聊 → none
只返回单个单词
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
    <title>层级RAG | 分层分块</title>
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
    <h2>🔥 层级RAG | 分层分块 + 多路召回 + 重排序</h2>
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
        # 层级RAG全链路
        candidates = hierarchical_retrieval(user_msg)
        context = rerank_candidates(user_msg, candidates)
    elif tool == "calc":
        context = calculate(user_msg)
    elif tool == "time":
        context = get_time()

    prompt = f"参考资料：{context}\n用户问题：{user_msg}，请依据资料简洁作答。"
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