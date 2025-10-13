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
completion = client.chat.completions.create(
    model="qwen3-coder-plus",  # 此处以DeepSeek-R1-671B为例，可按需更换模型名称。
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}],
)

print(completion.model_dump_json())