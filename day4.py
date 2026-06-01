# 第4天：提示词机器人（套万能公式）
print("=== 专业AI提示词生成器 ===")

def generate_prompt():
    # 输入角色
    role = input("1. 请输入AI角色：")
    # 输入任务
    task = input("2. 请输入任务：")
    # 输入要求
    req = input("3. 请输入要求（用逗号分隔）：")
    req_list = req.split("，")
    
    # 生成提示词
    prompt = f"""
你是{role}。
请完成任务：{task}
要求：
"""
    for i, r in enumerate(req_list, 1):
        prompt += f"{i}. {r}\n"
    
    print("\n===== 你的专业提示词已生成 =====")
    print(prompt)
    return prompt

# 启动
while True:
    print("\n输入 退出 结束")
    choice = input("开始生成提示词？(y/n)：")
    if choice == "退出" or choice == "n":
        break
    generate_prompt()

print("提示词学习完成！明天开始调用真正大模型API！")