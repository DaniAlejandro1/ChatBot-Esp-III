import os
from dotenv import load_dotenv

import openai
from typing import List, Dict
from .base import Provider


load_dotenv()

print("Cargando ChatGPT Provider...:  " + str(os.getenv('OPENAI_API_KEY')))

class ChatGPTProvider(Provider):
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "openai/gpt-4.1-mini"
    
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
            print(f"Error en ChatGPT: {e}")
            return "Error al procesar la consulta con ChatGPT"
    
    @property
    def name(self) -> str:
        return "ChatGPT"