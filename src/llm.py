from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import VLLM
import os
import requests
import json

class LLMModel:
    def __init__(model_name):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if model_name != "gpt3.5":
            self.llm = VLLM(
                    model="DeepSeek-Coder-33B",
                    trust_remote_code=True,
                    max_new_tokens=1500,
                    top_k=9,
                    top_p=0.95,
                    temperature=0.1,
                    tensor_parallel_size=2  # for distributed inference
                )
        else:
            self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    def complete(self, inputs, contexts):
        url = "https://api.chatanywhere.tech/v1/chat/completions"
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": self.format(inputs, contexts)
            }
        ]
        })
        headers = {
        'Authorization': f'Bearer {OPENAI_KEY}',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)

        return response.text
    
    def format(self, inputs, contexts):
        return f"Given following context: {contexts} and your need to complete following {inputs} in one line:"