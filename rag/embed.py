import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

class Embedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        
    def create_index(self, chunks_file: str = 'data/processed/chunks.parquet'):
        """Crea índice FAISS desde chunks"""
        # Cargar chunks
        self.chunks = pd.read_parquet(chunks_file)
        
        # Generar embeddings
        print("Generando embeddings...")
        texts = self.chunks['content'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Crear índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Índice creado con {len(texts)} documentos")
        
    def save_index(self, index_path: str = 'data/index.faiss', 
                   metadata_path: str = 'data/chunks_metadata.pkl'):
        """Guarda el índice y metadatos"""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        
        # Guardar metadatos
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks.to_dict('records'), f)
        
        print(f"Índice guardado en {index_path}")
        
    def load_index(self, index_path: str = 'data/index.faiss',
                   metadata_path: str = 'data/chunks_metadata.pkl'):
        """Carga índice y metadatos"""
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            chunks_data = pickle.load(f)
            self.chunks = pd.DataFrame(chunks_data)
        
        print(f"Índice cargado con {self.index.ntotal} vectores")
    
    def search(self, query: str, k: int = 4):
        """Busca documentos similares"""
        # Generar embedding de la consulta
        query_embedding = self.model.encode([query])
        
        # Buscar en FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Obtener documentos
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                chunk = self.chunks.iloc[idx].to_dict()
                results.append(chunk)
        
        return results

if __name__ == "__main__":
    embedder = Embedder()
    embedder.create_index()
    embedder.save_index()