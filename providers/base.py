from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Provider(ABC):
    """Clase base para proveedores LLM"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Envía mensajes al LLM y obtiene respuesta"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del proveedor"""
        pass