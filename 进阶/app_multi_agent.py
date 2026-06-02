# 模块五：多智能体 Multi-Agent
# 架构：规划Agent + 执行Agent(Function Calling) + 评审Agent
# 底层复用：Query Rewrite + 层级RAG + 多路召回 + Rerank
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
MAX_HISTORY_ROUND = 6
MAX_RETRY = 2  # 评审不通过最大重试执行次数

# ===================== 初始化基础组件 =====================
app = Flask(__name__)
zhipu_client = ZhipuAI(api_key=API_KEY)

# 向量库 & 嵌入模型
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

# 全局索引
all_fine_chunks = []
bm25_fine = None
coarse2fine_map = {}
rerank_model = CrossEncoder(RERANK_MODEL_PATH)

# ===================== 1. 分层RAG 索引初始化 =====================
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
结合上文补全指代、口语转为标准检索问句，仅输出改写后问句。
"""
    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ===================== 3. 底层工具 & RAG检索 =====================
def full_rag_search(query: str) -> str:
    # 粗块定位
    coarse_res = coarse_collection.query(query_texts=[query], n_results=TOP_K_COARSE)
    hit_coarse_ids = set()
    for meta in coarse_res["metadatas"][0]:
        if meta is not None and "id" in meta:
            hit_coarse_ids.add(meta["id"])

    candidate_fine_idx = set()
    for cid in hit_coarse_ids:
        if cid in coarse2fine_map:
            for idx in coarse2fine_map[cid]:
                candidate_fine_idx.add(idx)
    if not candidate_fine_idx:
        return "未查询到相关资料"

    # 多路召回
    candidates = set()
    fine_res = fine_collection.query(query_texts=[query], n_results=TOP_K_RECALL)
    for doc in fine_res["documents"][0]:
        candidates.add(doc)

    query_tokens = list(jieba.cut(query))
    bm25_scores = bm25_fine.get_scores(query_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:TOP_K_RECALL]
    for idx in top_bm25_idx:
        if idx in candidate_fine_idx:
            candidates.add(all_fine_chunks[idx])
    if not candidates:
        return "未查询到相关资料"

    # 重排序
    rank_pairs = [[query, cand] for cand in candidates]
    scores = rerank_model.predict(rank_pairs)
    sorted_items = sorted(zip(scores, candidates), reverse=True)
    top_texts = [item[1] for item in sorted_items[:TOP_K_RERANK]]
    return "\n---\n".join(top_texts)

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "表达式错误，无法计算"

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function Calling 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "full_rag_search",
            "description": "查询本地知识库文档资料",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "查询问题"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学四则运算",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统日期时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
tool_map = {
    "full_rag_search": full_rag_search,
    "calculator": calculator,
    "get_current_time": get_current_time
}

# ===================== 4. 三大智能体实现 =====================
# 4.1 规划Agent：任务拆解
def planner_agent(user_query: str) -> str:
    """接收需求，拆解为分步执行任务列表"""
    prompt = f"""
你是任务规划专家，请把用户需求拆解为有序执行任务列表。
规则：
1. 只输出清晰、可逐条执行的任务步骤
2. 不要多余解释、不要直接回答问题
3. 任务要简洁、顺序合理

用户需求：{user_query}
"""
    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# 4.2 执行Agent：Function Calling 执行任务
def executor_agent(task: str) -> str:
    """根据单条任务，调用工具执行"""
    msgs = [{"role": "user", "content": task}]
    max_call = 3
    for _ in range(max_call):
        resp = zhipu_client.chat.completions.create(
            model="glm-4-flash",
            messages=msgs,
            tools=tools
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        # 解析并执行工具
        for tool_call in msg.tool_calls:
            fname = tool_call.function.name
            fargs = json.loads(tool_call.function.arguments)
            func = tool_map[fname]
            if fname == "get_current_time":
                res = func()
            else:
                res = func(**fargs)
            msgs.append(msg.model_dump())
            msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fname,
                "content": res
            })
    final_resp = zhipu_client.chat.completions.create(model="glm-4-flash", messages=msgs)
    return final_resp.choices[0].message.content.strip()

# 4.3 评审Agent：结果校验
def reviewer_agent(original_query: str, task_result: str) -> (bool, str):
    """
    评审执行结果
    返回：(是否通过, 修正意见/最终内容)
    """
    prompt = f"""
你是结果评审员，负责检查回答质量。
原始需求：{original_query}
当前执行结果：{task_result}

评审规则：
1. 检查内容是否完整回答问题
2. 检查是否存在编造、幻觉、无效内容
3. 内容残缺、错误、答非所问 → 判定不通过，并给出修正要求
4. 内容合格 → 直接输出合格内容

输出格式：第一行只能写【通过】或【不通过】，第二行开始写内容/意见。
"""
    resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    content = resp.choices[0].message.content.strip()
    lines = content.split("\n", 1)
    status = lines[0].strip()
    detail = lines[1].strip() if len(lines) > 1 else ""
    return (status == "通过", detail)

# ===================== 5. 多Agent 总调度流程 =====================
def multi_agent_pipeline(user_input: str) -> str:
    # 1. 规划阶段
    task_list = planner_agent(user_input)
    print("📋 规划任务列表：\n", task_list)

    # 按行拆分多条任务
    tasks = [t.strip() for t in task_list.splitlines() if t.strip()]
    all_results = []

    for task in tasks:
        retry = 0
        while retry <= MAX_RETRY:
            # 2. 执行阶段
            exec_res = executor_agent(task)
            # 3. 评审阶段
            pass_flag, review_content = reviewer_agent(user_input, exec_res)
            if pass_flag:
                all_results.append(review_content)
                break
            else:
                retry += 1
                print(f"⚠️ 任务[{task}]评审不通过，重试 {retry}/{MAX_RETRY}")
        else:
            all_results.append(f"【任务执行失败】{task}")

    # 汇总所有任务结果
    total_prompt = f"""
用户原始问题：{user_input}
各任务执行结果汇总：
{"\n".join(all_results)}
请整合所有内容，给出通顺、完整的最终回答。
"""
    final_resp = zhipu_client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": total_prompt}]
    )
    return final_resp.choices[0].message.content.strip()

# ===================== 6. 对话记忆 =====================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "你是智能问答助手。"}]

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

messages = load_memory()

# ===================== 7. 网页界面 & 接口 =====================
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>多智能体 Multi-Agent</title>
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
    <h2>🔥 多智能体系统 | 规划+执行+评审</h2>
    <div class="chat-box" id="chat"></div>
    <div class="input-box">
        <input id="msg" placeholder="输入复杂问题..." autocomplete="off">
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
    # 问题改写
    rewrite_q = rewrite_query(user_raw, messages)
    # 多智能体全流程
    final_answer = multi_agent_pipeline(rewrite_q)
    # 保存记忆
    messages.append({"role": "user", "content": user_raw})
    messages.append({"role": "assistant", "content": final_answer})
    save_memory(messages)
    return jsonify({"reply": final_answer})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)