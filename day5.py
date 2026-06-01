# 第5天：智谱 GLM 聊天机器人（流式输出）
from zhipuai import ZhipuAI
import time

# ========== 把你的 API Key 填在这里 ==========
client = ZhipuAI(api_key="sk-3a1f5de582f745a782133ee7295560a4.SG5a20RYIDw2qnOB")

def ai_chat(message):
    # 调用智谱大模型
    response = client.chat.completions.create(
        model="glm-4-flash",  # 免费高速模型
        messages=[{"role": "user", "content": message}],
        stream=True,  # 流式逐字输出
    )

    print("AI：", end="")
    # 逐字打印效果
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            time.sleep(0.02)
    print("\n")

# 启动聊天
print("=== 智谱AI聊天机器人（第5天实战）===")
print("输入 退出 结束\n")

while True:
    user_input = input("你：")
    if user_input == "退出":
        print("再见！")
        break
    ai_chat(user_input)