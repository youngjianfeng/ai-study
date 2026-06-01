from zhipuai import ZhipuAI

# ===================== 配置区域 =====================
API_KEY = "sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB"
MODEL = "glm-4-flash"
# AI角色设定（可以随便改）
SYSTEM_PROMPT = """
你是一名专业的AI大模型应用开发助手，擅长：
1. 解答Python编程问题
2. 讲解大模型、RAG、Agent知识
3. 制定学习计划
4. 回答简洁、清晰、有耐心
回答不要太长，重点突出。
"""
# ======================================================

# 初始化客户端
client = ZhipuAI(api_key=API_KEY)

# 消息历史 = 记忆核心
messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# 项目标题
print("=" * 50)
print("    第7天 · 第一周终极项目：AI大模型助手")
print("=" * 50)
print("输入：退出 结束对话\n")

# 主对话循环
while True:
    user_input = input("你：")
    
    # 退出机制
    if user_input.strip() in ["退出", "exit", "quit"]:
        print("AI：很高兴为你服务，再见！")
        break
    
    # 把用户消息加入记忆
    messages.append({"role": "user", "content": user_input})

    # 调用大模型
    print("AI：", end="", flush=True)
    full_response = ""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        # 流式输出
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content

        # 把AI回复加入记忆
        messages.append({"role": "assistant", "content": full_response})
        print("\n")

    except Exception as e:
        print(f"\n出错啦：{e}")
        messages.pop()  # 出错时移除上一条消息