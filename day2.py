# 第2天实战：API 调用 + 联网AI工具
import requests

class WeatherAI:
    def __init__(self):
        self.name = "天气小助手"

    # 获取天气（模拟公开接口）
    def get_weather(self, city):
        # 这里用模拟数据，真实开发可换真实API
        return f"{city}：晴天，25℃，适合学习大模型"

    def get_clothes(self, weather):
        if "25℃" in weather:
            return "建议穿短袖+薄外套"
        else:
            return "正常穿着即可"

    # 聊天对话
    def chat(self):
        print(f"=== {self.name} 已启动 ===")
        while True:
            msg = input("\n你要查询哪个城市的天气？（退出=结束）：")
            if msg == "退出":
                print("再见！")
                break
            # 调用天气方法
            weather = self.get_weather(msg)
            print("weather：", weather)

            tips = self.get_clothes(weather)
            print("tips：", tips)

# 启动AI
if __name__ == "__main__":
    ai = WeatherAI()
    ai.chat()