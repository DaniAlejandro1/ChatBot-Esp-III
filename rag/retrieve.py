from .embed import Embedder
from typing import List, Dict

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self._load_index()
    
    def _load_index(self):
        """Intenta cargar el índice existente"""
        try:
            success = self.embedder.load_index()
            if success:
                print("✅ Índice FAISS cargado exitosamente")
            else:
                print("❌ No se pudo cargar el índice. Ejecuta 'python -m rag.embed' primero.")
        except Exception as e:
            print(f"❌ Error cargando índice: {e}")
    
    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Busca documentos relevantes para una consulta"""
        print(f"🔍 Buscando {k} documentos para: '{query}'")
        results = self.embedder.search(query, k)
        print(f"✅ Encontrados {len(results)} documentos relevantes")
        return results
    
    def get_retrieval_stats(self):
        """Retorna estadísticas del retriever"""
        return self.embedder.get_index_stats()