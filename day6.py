from zhipuai import ZhipuAI

# 填入你的智谱 API KEY
client = ZhipuAI(api_key="sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB")

# 全局消息列表 = 记忆核心
messages = [
    # 系统角色设定（最关键）
    {"role": "system", "content": "你是一位专业的AI大模型学习助手，温柔、耐心、专业，回答简洁易懂。"}
]

print("=== 第6天：带记忆的专业AI助手 ===")
print("输入 退出 结束\n")

while True:
    user_input = input("你：")
    if user_input == "退出":
        print("再见！")
        break

    # 把用户说的话加入记忆
    messages.append({"role": "user", "content": user_input})

    # 调用智谱大模型
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        stream=True
    )

    print("AI：", end="")
    full_answer = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_answer += content
    print()

    # 把AI的回答也加入记忆
    messages.append({"role": "assistant", "content": full_answer})