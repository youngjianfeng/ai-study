# 模块六：全离线私有化｜Ollama本地大模型 + 全链路RAG+MultiAgent
import os, json, datetime, jieba, requests
from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

# ==================== 全局配置 ====================
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b"
DOCS_FOLDER = "docs"
MEMORY_FILE = "memory_offline.json"
CHROMA_DB_PATH = "./chroma_offline_db"
COLLECTION_COARSE = "coarse_chunk"
COLLECTION_FINE = "fine_chunk"

#分块参数
COARSE_CHUNK_SIZE,COARSE_OVERLAP = 800,100
FINE_CHUNK_SIZE,FINE_OVERLAP = 300,50
TOP_K_COARSE=3
TOP_K_RECALL=8
TOP_K_RERANK=3
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_HISTORY_ROUND=6
MAX_RETRY=2

app=Flask(__name__)
#向量库初始化
chroma_client=chromadb.PersistentClient(path=CHROMA_DB_PATH)
emb_func=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
coarse_coll=chroma_client.get_or_create_collection(COLLECTION_COARSE,embedding_function=emb_func)
fine_coll=chroma_client.get_or_create_collection(COLLECTION_FINE,embedding_function=emb_func)
coarse_splitter=RecursiveCharacterTextSplitter(chunk_size=COARSE_CHUNK_SIZE,chunk_overlap=COARSE_OVERLAP)
fine_splitter=RecursiveCharacterTextSplitter(chunk_size=FINE_CHUNK_SIZE,chunk_overlap=FINE_OVERLAP)

all_fine_chunks=[]
bm25_fine=None
coarse2fine_map={}
rerank=CrossEncoder(RERANK_MODEL)

# ==================== 1、本地Ollama请求封装(替代zhipuAPI) ====================
def ollama_chat(sys_prompt:str,user_prompt:str)->str:
    payload={
        "model":MODEL_NAME,
        "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
        "stream":False
    }
    res=requests.post(OLLAMA_URL,json=payload)
    return res.json()["message"]["content"].strip()

# ==================== 2、构建分层索引 ====================
def init_index():
    global all_fine_chunks,bm25_fine,coarse2fine_map
    if coarse_coll.count()>0 and fine_coll.count()>0:
        print("✅ 加载已有分层索引")
        fine_data=fine_coll.get(include=["documents","metadatas"])
        all_fine_chunks=fine_data["documents"]
        for idx,meta in enumerate(fine_data["metadatas"]):
            cid=meta.get("coarse_id")
            if cid not in coarse2fine_map:coarse2fine_map[cid]=[]
            coarse2fine_map[cid].append(idx)
        corpus=[list(jieba.cut(d)) for d in all_fine_chunks]
        bm25_fine=BM25Okapi(corpus)
        return
    print("⏳ 构建离线知识库索引")
    coarse_id,fine_id=0,0
    for fname in os.listdir(DOCS_FOLDER):
        fp=os.path.join(DOCS_FOLDER,fname)
        try:
            if fname.endswith(".txt"):loader=TextLoader(fp,encoding="utf-8")
            elif fname.endswith(".pdf"):loader=PyPDFLoader(fp)
            elif fname.endswith(".docx"):loader=Docx2txtLoader(fp)
            else:continue
            docs=loader.load()
            coarse_list=coarse_splitter.split_documents(docs)
            for ck in coarse_list:
                cid=f"coarse_{coarse_id}"
                coarse_id+=1
                ctxt=ck.page_content
                coarse_coll.add(documents=[ctxt],ids=[cid])
                fine_list=fine_splitter.split_text(ctxt)
                for ftxt in fine_list:
                    fid=f"fine_{fine_id}"
                    fine_id+=1
                    all_fine_chunks.append(ftxt)
                    fine_coll.add(documents=[ftxt],ids=[fid],metadatas=[{"coarse_id":cid}])
                    if cid not in coarse2fine_map:coarse2fine_map[cid]=[]
                    coarse2fine_map[cid].append(len(all_fine_chunks)-1)
        except Exception as e:
            print(f"跳过:{fname} {e}")
    corpus=[list(jieba.cut(d)) for d in all_fine_chunks]
    bm25_fine=BM25Okapi(corpus)
    print("✅ 索引构建完成")
init_index()

# ====================3、RAG检索全链路 ====================
def full_rag_search(q:str)->str:
    # 强制转字符串，防止传列表
    q = str(q).strip()
    #粗块定位
    coarse_res=coarse_coll.query([q],n_results=TOP_K_COARSE)
    hit_cid={m["id"] for m in coarse_res["metadatas"][0]}
    fine_idx=set()
    for c in hit_cid:
        if c in coarse2fine_map:fine_idx.update(coarse2fine_map[c])
    if not fine_idx:return "未检索到相关资料"
    #多路召回
    cand=set()
    vec_res=fine_coll.query([q],n_results=TOP_K_RECALL)
    [cand.add(d) for d in vec_res["documents"][0]]
    tokens=list(jieba.cut(q))
    bm_score=bm25_fine.get_scores(tokens)
    top_bm=sorted(range(len(bm_score)),key=lambda x:bm_score[x],reverse=True)[:TOP_K_RECALL]
    [cand.add(all_fine_chunks[i]) for i in top_bm if i in fine_idx]
    if not cand:return "未检索到相关资料"
    #重排序
    pairs=[[q,item] for item in cand]
    score=rerank.predict(pairs)
    sort_data=sorted(zip(score,cand),reverse=True)
    top=[i[1] for i in sort_data[:TOP_K_RERANK]]
    return "\n---\n".join(top)

#工具函数
def calc(expr:str):
    try:return str(eval(expr))
    except:return "表达式错误"
def now_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#FunctionCalling工具定义
tools=[
    {"type":"function","function":{"name":"full_rag_search","description":"查询本地知识库文档","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"calc","description":"数学运算","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}},
    {"type":"function","function":{"name":"now_time","description":"获取当前系统时间","parameters":{}}}
]
tool_map={"full_rag_search":full_rag_search,"calc":calc,"now_time":now_time}

# ====================4、QueryRewrite ====================
def rewrite_query(raw_q:str,history:list)->str:
    recent=history[-MAX_HISTORY_ROUND:]
    his_str=""
    for m in recent:
        if m["role"] in ("user","assistant"):his_str+=f"{m['role']}:{m['content']}\n"
    sys="你是问句改写助手，结合历史补全指代、口语转标准检索句，只输出改写后的一句话，无多余内容"
    user=f"{his_str}\n原问题:{raw_q}"
    return ollama_chat(sys,user)

# ====================5、三大智能体(规划/执行/评审) ====================
def planner_agent(q:str)->str:
    sys="任务拆分专家，只输出有序任务列表，不回答问题、不加多余文字"
    user=f"用户需求:{q}"
    return ollama_chat(sys,user)

def exec_agent(task:str)->str:
    #本地简易FC：ollama无原生FC，采用提示词模拟工具调用
    sys=f"""可用工具：
1.full_rag_search(query):查知识库，参数只传纯文本
2.calc(expr):数学计算
3.now_time():获取时间
需要工具则严格输出：工具名|参数，**参数不要加[]、""、''，只输出原始文字，不要任何符号**，不需要工具直接输出答案"""
    res=ollama_chat(sys,task).strip()
    if "|" in res and res.count("|")==1:
        name,arg=res.split("|")
        # 1.清理工具名多余符号【】空格括号
        name = re.sub(r'[【】\[\]"\'{}（） ]','',name.strip())
        # 2.清理参数的 [] "" '' 等包裹符号
        arg = re.sub(r'^[\[\'"]+|[\]\'"]+$','',arg.strip())

        if name=="now_time":
            return tool_map[name]()
        else:
            # arg此时是干净字符串
            return tool_map[name](arg)
    return res

def review_agent(origin_q:str,res:str)->(bool,str):
    sys="评审结果，首行只写【通过】/【不通过】，换行写内容或整改意见"
    user=f"需求:{origin_q}\n结果:{res}"
    ans=ollama_chat(sys,user)
    sp=ans.split("\n",1)
    flag=sp[0].strip()=="通过"
    cnt=sp[1].strip() if len(sp)>1 else ""
    return flag,cnt

#多Agent总流程
def multi_agent_run(q:str)->str:
    task_txt=planner_agent(q)
    task_list=[i.strip() for i in task_txt.splitlines() if i.strip()]
    all_res=[]
    for t in task_list:
        rt=0
        ans=""
        while rt<=MAX_RETRY:
            ans=exec_agent(t)
            ok,detail=review_agent(q,ans)
            if ok:
                all_res.append(detail)
                break
            rt+=1
        else:all_res.append(f"任务失败:{t}")
    #汇总输出
    sys="整合多条结果，通顺完整回答用户原始问题"
    user=f"原问题:{q}\n分项结果:\n{all_res}"
    return ollama_chat(sys,user)

# ====================记忆&前端 ====================
def load_mem():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE,"r",encoding="utf-8")as f:
            return json.load(f)
    return [{"role":"system","content":"离线知识库助手，依据本地资料回答"}]
def save_mem(data):
    with open(MEMORY_FILE,"w",encoding="utf-8")as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
mem=load_mem()

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
<meta charset=utf-8>
<style>body{max-width:750px;margin:30px auto;font-family:Arial}.chat-box{height:520px;overflow-y:auto;border:1px solid #ddd;padding:16px;border-radius:8px;background:#f9f9f9}.msg{margin:8px 0;padding:10px 14px;border-radius:8px;max-width:72%}.user{background:#007bff;color:#fff;margin-left:auto}.bot{background:#e6e6e6;color:#222;margin-right:auto}.inp{display:flex;margin-top:12px}input{flex:1;padding:12px;border:1px solid #ddd;border-radius:6px}button{padding:12px 18px;background:#007bff;color:#fff;border:0;border-radius:6px;margin-left:8px}</style>
</head>
<body>
<h3>🔥全离线Ollama知识库｜多Agent</h3>
<div class="chat-box" id="chat"></div>
<div class="inp"><input id="msg" placeholder="输入问题"><button onclick="send()">发送</button></div>
<script>
function add(txt,isUser){let d=document.createElement("div");d.className="msg "+(isUser?"user":"bot");d.innerText=txt;chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function send(){let v=msg.value.trim();if(!v)return;add(v,1);msg.value="";let d=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({msg:v})});let r=await d.json();add(r.reply,0)}
</script>
</body></html>'''

@app.route('/chat',methods=["POST"])
def chat():
    raw=request.json.get("msg")
    rewrite_q=rewrite_query(raw,mem)
    final=multi_agent_run(rewrite_q)
    mem.append({"role":"user","content":raw})
    mem.append({"role":"assistant","content":final})
    save_mem(mem)
    return jsonify({"reply":final})

if __name__=="__main__":
    app.run(port=5000,debug=True)