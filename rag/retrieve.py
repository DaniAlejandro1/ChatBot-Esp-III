from .embed import Embedder
from typing import List, Dict

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        try:
            self.embedder.load_index()
        except:
            print("Índice no encontrado. Ejecute primero rag/embed.py")
    
    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Busca documentos relevantes para una consulta"""
        return self.embedder.search(query, k)
    
    def rerank(self, docs: List[Dict], query: str) -> List[Dict]:
        """Re-rankea documentos (opcional - implementación simple)"""
        # Por ahora retorna los mismos documentos
        # Se puede implementar un reranker más sofisticado
        return docs