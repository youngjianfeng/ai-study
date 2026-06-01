# 打印输出
print("你好，AI大模型开发！")

# 变量存储数据
name = "小明"
age = 25
print(name, age)

# 列表：存一组数据
tools = ["ChatGPT", "通义千问", "LangChain"]
print(tools[0])  # 取第一个

# 字典：存键值对（AI开发超级常用）
user_info = {"name": "AI助手", "model": "GPT-4"}
print(user_info["name"])

score = 90
if score >= 60:
    print("及格")
else:
    print("不及格")

# 遍历列表
for tool in tools:
    print("AI工具：", tool)

# 定义函数
def say_hello(name):
    return f"你好，{name}！"

# 调用函数
print(say_hello("AI工程师"))

# 获取用户输入
user_input = input("请输入你的问题：")
print("你输入的是：", user_input)

# 第1天实战：简易AI对话机器人
print("=== 我的第一个AI助手 ===")

# 定义回答函数
def ai_answer(question):
    # 简单规则匹配（后面会换成真正大模型）
    if "你好" in question:
        return "你好呀！我是你的AI学习助手"
    elif "名字" in question:
        return "我叫AI小助手"
    elif "学习" in question:
        return "我们一起学习大模型开发！"
    else:
        return "我正在学习，明天就能更聪明啦！"

# 循环对话
while True:
    question = input("\n你：")
    if question == "退出":
        print("再见！")
        break
    answer = ai_answer(question)
    print("AI：", answer)