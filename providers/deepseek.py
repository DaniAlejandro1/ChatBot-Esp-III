import os
from openai import OpenAI
from typing import List, Dict
from .base import Provider

class DeepSeekProvider(Provider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = "deepseek-chat"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 500)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error en DeepSeek: {e}")
            return "Error al procesar la consulta con DeepSeek"
    
    @property
    def name(self) -> str:
        return "DeepSeek"