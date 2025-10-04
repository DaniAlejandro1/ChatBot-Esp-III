import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import os
from typing import List, Dict

class Embedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.metadata_map = {}
        
    def create_index(self, chunks_file: str = 'data/processed/chunks.parquet'):
        """Crea índice FAISS desde chunks"""
        # Cargar chunks
        self.chunks = pd.read_parquet(chunks_file)
        
        # Verificar metadatos requeridos
        required_columns = ['content', 'doc_id', 'title', 'page', 'source']
        missing = [col for col in required_columns if col not in self.chunks.columns]
        if missing:
            print(f"⚠️  Faltan columnas: {missing}. Usando columnas disponibles: {self.chunks.columns.tolist()}")
        
        print("Generando embeddings...")
        texts = self.chunks['content'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Crear índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para cosine similarity
        faiss.normalize_L2(embeddings)  # Normalizar para cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        # Construir mapa de metadatos
        for idx, chunk in self.chunks.iterrows():
            self.metadata_map[idx] = {
                'content': chunk['content'],
                'doc_id': chunk.get('doc_id', 'unknown'),
                'title': chunk.get('title', 'Unknown'),
                'page': chunk.get('page', 1),
                'source': chunk.get('source', 'unknown'),
                'chunk_id': chunk.get('chunk_id', f"chunk_{idx}")
            }
        
        print(f"✅ Índice creado: {len(texts)} chunks, {dimension} dimensiones")
        
    def save_index(self, index_path: str = 'data/index.faiss', 
                   metadata_path: str = 'data/chunks_metadata.pkl'):
        """Guarda el índice y metadatos"""
        try:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.index is None:
                print("❌ No hay índice para guardar. Ejecuta create_index primero.")
                return False
            
            # Guardar índice FAISS
            faiss.write_index(self.index, index_path)
            
            # Guardar metadatos
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks_data': self.chunks.to_dict('records'),
                    'metadata_map': self.metadata_map
                }, f)
            
            print(f"✅ Índice guardado en {index_path}")
            print(f"✅ Metadatos guardados en {metadata_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando índice: {e}")
            return False
        
    def load_index(self, index_path: str = 'data/index.faiss',
                   metadata_path: str = 'data/chunks_metadata.pkl'):
        """Carga índice y metadatos"""
        try:
            if not os.path.exists(index_path):
                print(f"❌ Archivo de índice no encontrado: {index_path}")
                return False
            
            self.index = faiss.read_index(index_path)
            
            with open(metadata_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.chunks = pd.DataFrame(saved_data['chunks_data'])
                self.metadata_map = saved_data['metadata_map']
            
            print(f"✅ Índice cargado: {self.index.ntotal} vectores")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando índice: {e}")
            return False
    
    def search(self, query: str, k: int = 4):
        """Busca documentos similares"""
        if self.index is None:
            print("❌ Índice no cargado. Ejecuta load_index primero.")
            return []
        
        # Generar embedding de consulta
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Búsqueda con scores
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Filtrar y ordenar resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata_map) and score > 0.1:  # Umbral de relevancia
                metadata = self.metadata_map.get(idx, {})
                results.append({
                    'content': metadata.get('content', ''),
                    'doc_id': metadata.get('doc_id', ''),
                    'title': metadata.get('title', ''),
                    'page': metadata.get('page', 1),
                    'source': metadata.get('source', ''),
                    'score': float(score),
                    'chunk_id': metadata.get('chunk_id', '')
                })
        
        # Ordenar por score descendente
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def get_index_stats(self):
        """Retorna estadísticas del índice"""
        if self.index is None:
            return {"status": "No cargado"}
        
        return {
            "status": "Cargado",
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self.chunks) if self.chunks is not None else 0,
            "dimension": self.index.d if hasattr(self.index, 'd') else "Unknown"
        }

def main():
    """Función principal para generar embeddings"""
    embedder = Embedder()
    
    print("🚀 Iniciando generación de embeddings...")
    
    # Crear índice
    embedder.create_index()
    
    # Guardar índice
    success = embedder.save_index()
    
    if success:
        print("🎉 Proceso completado exitosamente!")
        
        # Mostrar estadísticas
        stats = embedder.get_index_stats()
        print(f"📊 Estadísticas del índice:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Error en el proceso")

if __name__ == "__main__":
    main()