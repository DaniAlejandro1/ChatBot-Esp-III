import os
import json
import pandas as pd
import re  # Asegúrate de importar re
from typing import List, Dict
import PyPDF2
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extrae texto de un PDF y lo divide en chunks"""
    chunks = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if not text or text.strip() == "":
                    continue
                
                # CORRECCIÓN: usar re.sub en lugar de re.subs
                text = re.sub(r'\n+', ' ', text)  # Unificar saltos de línea
                text = re.sub(r'\s+', ' ', text)  # Unificar espacios
                text = text.strip()
                
                # Si el texto es muy corto o parece corrupto, saltar
                if len(text) < 50:
                    continue
                
                # Dividir en chunks más simples
                sentences = re.split(r'[.!?]+', text)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk + " " + sentence) < 800:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if len(current_chunk) > 50:
                            chunks.append({
                                'content': current_chunk,
                                'doc_id': os.path.basename(pdf_path),
                                'title': os.path.basename(pdf_path).replace('.pdf', ''),
                                'page': page_num + 1,
                                'source': pdf_path,
                                'chunk_id': f"{os.path.basename(pdf_path)}_p{page_num+1}_c{len(chunks)}"
                            })
                        current_chunk = sentence
                
                # Último chunk de la página
                if current_chunk and len(current_chunk) > 50:
                    chunks.append({
                        'content': current_chunk,
                        'doc_id': os.path.basename(pdf_path),
                        'title': os.path.basename(pdf_path).replace('.pdf', ''),
                        'page': page_num + 1,
                        'source': pdf_path,
                        'chunk_id': f"{os.path.basename(pdf_path)}_p{page_num+1}_c{len(chunks)}"
                    })
    
    except Exception as e:
        print(f"❌ Error procesando {pdf_path}: {e}")
        # Continuar con otros archivos
    
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
        
        if chunks:
            all_chunks.extend(chunks)
            print(f"  ✅ Extraídos {len(chunks)} chunks")
            
            # Registrar fuente
            sources.append({
                'doc_id': pdf_file.name,
                'title': pdf_file.stem,
                'source': str(pdf_file),
                'vigencia': '2024-2025',
                'url': 'https://www.ufro.cl/normativa'
            })
        else:
            print(f"  ⚠️  No se pudieron extraer chunks de {pdf_file}")
    
    if all_chunks:
        # Guardar chunks
        df_chunks = pd.DataFrame(all_chunks)
        df_chunks.to_parquet(f"{output_dir}/chunks.parquet")
        
        # Guardar fuentes
        df_sources = pd.DataFrame(sources)
        df_sources.to_csv(f"{output_dir}/sources.csv", index=False)
        
        print(f"✅ Procesados {len(all_chunks)} chunks de {len(sources)} documentos")
    else:
        print("❌ No se pudieron procesar chunks de ningún documento")
        # Crear chunks de ejemplo para que el sistema funcione
        create_sample_chunks(output_dir)

def create_sample_chunks(output_dir: str):
    """Crea chunks de ejemplo si no hay PDFs procesables"""
    sample_chunks = [
        {
            'content': 'El reglamento de convivencia universitaria establece normas de conducta para estudiantes y funcionarios de la UFRO.',
            'doc_id': 'reglamento_convivencia.pdf',
            'title': 'Reglamento de Convivencia Universitaria',
            'page': 1,
            'source': 'data/raw/reglamento_convivencia.pdf',
            'chunk_id': 'reglamento_convivencia_p1_c1'
        },
        {
            'content': 'La asistencia mínima requerida para actividades teóricas es del 75% según el reglamento académico.',
            'doc_id': 'reglamento_academico.pdf', 
            'title': 'Reglamento Académico',
            'page': 12,
            'source': 'data/raw/reglamento_academico.pdf',
            'chunk_id': 'reglamento_academico_p12_c1'
        },
        {
            'content': 'Los estudiantes pueden inscribir máximo 30 créditos por semestre según la normativa vigente.',
            'doc_id': 'reglamento_academico.pdf',
            'title': 'Reglamento Académico', 
            'page': 8,
            'source': 'data/raw/reglamento_academico.pdf',
            'chunk_id': 'reglamento_academico_p8_c1'
        },
        {
            'content': 'El plazo para apelar calificaciones es de 5 días hábiles desde su publicación oficial.',
            'doc_id': 'reglamento_evaluacion.pdf',
            'title': 'Reglamento de Evaluación',
            'page': 15,
            'source': 'data/raw/reglamento_evaluacion.pdf',
            'chunk_id': 'reglamento_evaluacion_p15_c1'
        }
    ]
    
    df_chunks = pd.DataFrame(sample_chunks)
    df_chunks.to_parquet(f"{output_dir}/chunks.parquet")
    
    sources = [
        {
            'doc_id': 'reglamento_convivencia.pdf',
            'title': 'Reglamento de Convivencia Universitaria', 
            'source': 'data/raw/reglamento_convivencia.pdf',
            'vigencia': '2024-2025',
            'url': 'https://www.ufro.cl/normativa/convivencia'
        },
        {
            'doc_id': 'reglamento_academico.pdf',
            'title': 'Reglamento Académico',
            'source': 'data/raw/reglamento_academico.pdf', 
            'vigencia': '2024-2025',
            'url': 'https://www.ufro.cl/normativa/academico'
        },
        {
            'doc_id': 'reglamento_evaluacion.pdf',
            'title': 'Reglamento de Evaluación',
            'source': 'data/raw/reglamento_evaluacion.pdf',
            'vigencia': '2024-2025',
            'url': 'https://www.ufro.cl/normativa/evaluacion'
        }
    ]
    
    df_sources = pd.DataFrame(sources)
    df_sources.to_csv(f"{output_dir}/sources.csv", index=False)
    
    print("✅ Creados chunks de ejemplo para continuar con el desarrollo")
    print("📝 NOTA: Reemplaza estos chunks cuando tengas PDFs reales procesables")

if __name__ == "__main__":
    process_documents()