import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件到环境变量
load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用ideaLAB的API Key将下行替换为：api_key="xxx",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
messages = [
    {"role": "system", "content": "你是一位历史研究专家。"},
    {"role": "user", "content": "三大战役各个军队伤亡如何？"}
]
completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=messages,
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    extra_body={"enable_thinking": True},
)
print(completion)
messages.append({
    "role": "assistant",
    "content": "<think>" + completion.choices[0].message.content + "</think>\n" + completion.choices[0].message.content
})

print(messages)

