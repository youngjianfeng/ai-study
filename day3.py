# day3.py：Git + Linux 命令学习工具
print("=== AI 命令助手 ===")

def show_commands():
    print("""
常用命令：
1. ls    查看文件
2. cd    进入文件夹
3. mkdir 创建文件夹
4. git add .   添加代码
5. git commit  提交版本
""")

while True:
    ipt = input("\n输入命令（输入 help 查看，退出=结束）：")
    if ipt == "退出":
        print("再见！")
        break
    elif ipt == "help":
        show_commands()
    else:
        print(f"你输入了：{ipt}，继续练习吧！")