import os
import json
import pandas as pd
from typing import List, Dict
import PyPDF2
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extrae texto de un PDF y lo divide en chunks"""
    chunks = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # Limpiar texto
            text = text.replace('\n\n', '\n').strip()
            
            # Dividir en chunks de ~900 tokens
            words = text.split()
            chunk_size = 200  # ~900 caracteres
            overlap = 20  # ~90 caracteres de solapamiento
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text) > 50:  # Filtrar chunks muy pequeños
                    chunks.append({
                        'content': chunk_text,
                        'doc_id': os.path.basename(pdf_path),
                        'title': os.path.basename(pdf_path).replace('.pdf', ''),
                        'page': page_num + 1,
                        'source': pdf_path
                    })
    
    return chunks

def process_documents(input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
    """Procesa todos los documentos PDF"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    sources = []
    
    # Procesar cada PDF
    for pdf_file in Path(input_dir).glob('*.pdf'):
        print(f"Procesando: {pdf_file}")
        chunks = extract_text_from_pdf(str(pdf_file))
        all_chunks.extend(chunks)
        
        # Registrar fuente
        sources.append({
            'doc_id': pdf_file.name,
            'title': pdf_file.stem,
            'source': str(pdf_file),
            'vigencia': '2024-2025',
            'url': 'https://www.ufro.cl/normativa'  # Placeholder
        })
    
    # Guardar chunks
    df_chunks = pd.DataFrame(all_chunks)
    df_chunks.to_parquet(f"{output_dir}/chunks.parquet")
    
    # Guardar fuentes
    df_sources = pd.DataFrame(sources)
    df_sources.to_csv(f"{output_dir}/sources.csv", index=False)
    
    print(f"Procesados {len(all_chunks)} chunks de {len(sources)} documentos")
    return all_chunks

if __name__ == "__main__":
    process_documents()