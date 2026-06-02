# 模块四：原生 Function Calling 函数调用
# 全链路：Query Rewrite + 层级RAG + 多路召回 + Rerank + 原生函数调用Agent
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
COLLECTION_COARSE = "coarse_chunk"
COLLECTION_FINE = "fine_chunk"

# 分块参数
COARSE_CHUNK_SIZE = 800
COARSE_OVERLAP = 100
FINE_CHUNK_SIZE = 300
FINE_OVERLAP = 50

# 检索&重排序参数
TOP_K_COARSE = 3
TOP_K_RECALL = 8
TOP_K_RERANK = 3
RERANK_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 对话历史限制
MAX_HISTORY_ROUND = 6

# ===================== 初始化基础组件 =====================
app = Flask(__name__)
zhipu_client = ZhipuAI(api_key=API_KEY)

# 向量库与嵌入模型
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
coarse_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_COARSE, embedding_function=embedding_func
)
fine_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_FINE, embedding_function=embedding_func
)

# 分块器
coarse_splitter = RecursiveCharacterTextSplitter(
    chunk_size=COARSE_CHUNK_SIZE, chunk_overlap=COARSE_OVERLAP
)
fine_splitter = RecursiveCharacterTextSplitter(
    chunk_size=FINE_CHUNK_SIZE, chunk_overlap=FINE_OVERLAP
)

# 全局索引变量
all_fine_chunks = []
bm25_fine = None
coarse2fine_map = {}

# 重排序模型
rerank_model = CrossEncoder(RERANK_MODEL_PATH)

# ===================== 1. 分层索引初始化 =====================
def init_hierarchical_index():
    global all_fine_chunks, bm25_fine, coarse2fine_map
    if coarse_collection.count() > 0 and fine_collection.count() > 0:
        print("✅ 分层索引已加载")
        fine_data = fine_collection.get(include=["documents", "metadatas"])
        all_fine_chunks = fine_data["documents"]
        for idx, meta in enumerate(fine_data["metadatas"]):
            cid = meta.get("coarse_id")
            if cid not in coarse2fine_map:
                coarse2fine_map[cid] = []
            coarse2fine_map[cid].append(idx)
        token_corpus = [list(jieba.cut(doc)) for doc in all_fine_chunks]
        bm25_fine = BM25Okapi(token_corpus)
        return

    print("⏳ 构建分层索引...")
    coarse_id_counter = 0
    fine_id_counter = 0
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
            coarse_chunks = coarse_splitter.split_documents(docs)
            for coarse_chunk in coarse_chunks:
                c_id = f"coarse_{coarse_id_counter}"
                coarse_id_counter += 1
                c_text = coarse_chunk.page_content
                coarse_collection.add(documents=[c_text], ids=[c_id])

                fine_sub_chunks = fine_splitter.split_text(c_text)
                for f_text in fine_sub_chunks:
                    f_id = f"fine_{fine_id_counter}"
                    fine_id_counter += 1
                    all_fine_chunks.append(f_text)
                    fine_collection.add(
                        documents=[f_text],
                        ids=[f_id],
                        metadatas=[{"coarse_id": c_id}]
                    )
                    if c_id not in coarse2fine_map:
                        coarse2fine_map[c_id] = []
                    coarse2fine_map[c_id].append(len(all_fine_chunks)-1)
        except Exception as e:
            print(f"❌ 跳过 {filename}：{str(e)}")

    token_corpus = [list(jieba.cut(doc)) for doc in all_fine_chunks]
    bm25_fine = BM25Okapi(token_corpus)
    print("✅ 分层索引构建完成")

init_hierarchical_index()

# ===================== 2. Query Rewrite 问题改写 =====================
def rewrite_query(raw_query: str, history_msgs: list) -> str:
    recent_history = history_msgs[-MAX_HISTORY_ROUND:]
    history_str = ""
    for msg in recent_history:
        if msg["role"] in ("user", "assistant"):
            history_str += f"{msg['role']}：{msg['content']}\n"

    prompt = f"""
历史对话：
{history_str}
当前用户新问题：{raw_query}

任务：
1. 结合上文补全指代、省略内容
2. 口语转标准书面检索问句
3. 仅输出改写后的问句，无额外文字
"""
    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ===================== 3. RAG 检索全链路 =====================
def full_rag_search(query: str) -> str:
    """层级RAG + 多路召回 + 重排序"""
    # 1. 粗块定位
    coarse_res = coarse_collection.query(query_texts=[query], n_results=TOP_K_COARSE)
    hit_coarse_ids = set()
    for meta in coarse_res["metadatas"][0]:
        if meta is not None and "id" in meta:
            hit_coarse_ids.add(meta["id"])

    # 2. 圈定细块范围
    candidate_fine_idx = set()
    for cid in hit_coarse_ids:
        if cid in coarse2fine_map:
            for idx in coarse2fine_map[cid]:
                candidate_fine_idx.add(idx)
    if not candidate_fine_idx:
        return "未查询到相关资料"

    # 3. 多路召回
    candidates = set()
    # 语义召回
    fine_res = fine_collection.query(query_texts=[query], n_results=TOP_K_RECALL)
    for doc in fine_res["documents"][0]:
        candidates.add(doc)
    # BM25关键词召回
    query_tokens = list(jieba.cut(query))
    bm25_scores = bm25_fine.get_scores(query_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:TOP_K_RECALL]
    for idx in top_bm25_idx:
        if idx in candidate_fine_idx:
            candidates.add(all_fine_chunks[idx])
    if not candidates:
        return "未查询到相关资料"

    # 4. 重排序
    rank_pairs = [[query, cand] for cand in candidates]
    scores = rerank_model.predict(rank_pairs)
    sorted_items = sorted(zip(scores, candidates), reverse=True)
    top_texts = [item[1] for item in sorted_items[:TOP_K_RERANK]]
    return "\n---\n".join(top_texts)

# ===================== 4. 工具函数定义 =====================
def calculator(expression: str) -> str:
    """数学计算工具"""
    try:
        return str(eval(expression))
    except Exception:
        return "表达式错误，无法计算"

def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================== 5. 定义工具描述（Function Calling 核心） =====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "full_rag_search",
            "description": "查询本地知识库，获取文档、资料、相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查询的问题"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学四则运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 1+2*3、(10-5)/2"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统日期和时间",
            "parameters": {}
        }
    }
]

# 工具映射：名称 -> 函数
tool_map = {
    "full_rag_search": full_rag_search,
    "calculator": calculator,
    "get_current_time": get_current_time
}

# ===================== 6. 记忆读写 =====================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是全能AI助手，根据工具结果如实回答。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# ===================== 7. Function Calling 核心调度逻辑 =====================
def run_agent(user_input: str) -> str:
    """
    完整Agent调度：模型决策 -> 调用工具 -> 结果汇总
    支持多轮工具调用
    """
    current_msgs = messages.copy()
    current_msgs.append({"role": "user", "content": user_input})

    # 限制最大调用轮数，防止死循环
    max_call_round = 3
    for _ in range(max_call_round):
        # 调用模型，传入工具列表
        resp = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=current_msgs,
            tools=tools
        )
        msg = resp.choices[0].message
        # 场景1：不需要调用工具，直接返回回答
        if not msg.tool_calls:
            return msg.content
        # 场景2：解析工具调用
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            # 执行工具
            func = tool_map[func_name]
            if func_name == "get_current_time":
                tool_result = func()
            else:
                tool_result = func(**func_args)
            # 把工具调用记录 + 结果 追加到上下文
            current_msgs.append(msg.model_dump())
            current_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": tool_result
            })
    # 多轮调用结束，最后一次生成回答
    final_resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=current_msgs
    )
    return final_resp.choices[0].message.content

# ===================== 8. 网页界面 & 接口 =====================
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Function Calling 高阶Agent</title>
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
    <h2>🔥 高阶Agent | 原生Function Calling + 全链路RAG</h2>
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
    user_raw = request.json.get("msg")
    # 1. 问题改写
    rewrite_q = rewrite_query(user_raw, messages)
    # 2. Agent 调度执行
    reply = run_agent(rewrite_q)
    # 3. 保存原始对话
    messages.append({"role": "user", "content": user_raw})
    messages.append({"role": "assistant", "content": reply})
    save_memory(messages)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)