import os
import sys
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

sys.stdout.reconfigure(encoding='utf-8')


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token"
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("what is gen ai?")

print(response.content)