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

def extract_text_from_txt(txt_path: str) -> List[Dict]:
    """Extrae texto de archivos TXT y Markdown"""
    chunks = []
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if not content.strip():
            print(f"⚠️  Archivo vacío: {txt_path}")
            return chunks
        
        # Limpiar contenido de markdown básico
        import re
        cleaned_content = re.sub(r'#{1,6}\s*', '', content)  # Remover headers #
        cleaned_content = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_content)  # Remover **bold**
        cleaned_content = re.sub(r'\*(.*?)\*', r'\1', cleaned_content)  # Remover *italic*
        cleaned_content = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_content)  # Remover imágenes
        cleaned_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned_content)  # Remover links pero mantener texto
        
        # Dividir en párrafos naturales
        paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 100:  # Filtrar párrafos muy cortos
                chunks.append({
                    'content': paragraph,
                    'doc_id': os.path.basename(txt_path),
                    'title': os.path.basename(txt_path).replace('.txt', '').replace('.md', ''),
                    'page': i + 1,  # Usar número de párrafo como "página"
                    'source': txt_path,
                    'chunk_id': f"{os.path.basename(txt_path)}_p{i+1}"
                })
        
        print(f"  ✅ TXT procesado: {len(chunks)} chunks de {len(paragraphs)} párrafos")
        return chunks
        
    except Exception as e:
        print(f"❌ Error procesando TXT {txt_path}: {e}")
        return chunks

def process_documents(input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
    """Procesa todos los documentos (PDF, TXT, MD)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    sources = []
    
    print("📁 BUSCANDO DOCUMENTOS...")
    
    # Procesar PDFs
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    print(f"📄 PDFs encontrados: {len(pdf_files)}")
    
    for pdf_file in pdf_files:
        print(f"  🔍 Procesando PDF: {pdf_file.name}")
        chunks = extract_text_from_pdf(str(pdf_file))
        if chunks:
            all_chunks.extend(chunks)
            sources.append({
                'doc_id': pdf_file.name,
                'title': pdf_file.stem,
                'source': str(pdf_file),
                'vigencia': '2024-2025',
                'url': 'https://www.ufro.cl/normativa',
                'formato': 'pdf'
            })
            print(f"    ✅ {len(chunks)} chunks extraídos")
        else:
            print(f"    ⚠️  No se pudieron extraer chunks")
    
    # Procesar TXT y Markdown
    txt_files = list(Path(input_dir).glob('*.txt')) + list(Path(input_dir).glob('*.md'))
    print(f"📝 TXT/MD encontrados: {len(txt_files)}")
    
    for txt_file in txt_files:
        print(f"  🔍 Procesando TXT/MD: {txt_file.name}")
        chunks = extract_text_from_txt(str(txt_file))
        if chunks:
            all_chunks.extend(chunks)
            sources.append({
                'doc_id': txt_file.name,
                'title': txt_file.stem,
                'source': str(txt_file),
                'vigencia': '2024-2025', 
                'url': 'https://www.ufro.cl/normativa',
                'formato': 'txt'
            })
            print(f"    ✅ {len(chunks)} chunks extraídos")
        else:
            print(f"    ⚠️  No se pudieron extraer chunks")
    
    if all_chunks:
        # Guardar chunks
        df_chunks = pd.DataFrame(all_chunks)
        df_chunks.to_parquet(f"{output_dir}/chunks.parquet")
        
        # Guardar fuentes
        df_sources = pd.DataFrame(sources)
        df_sources.to_csv(f"{output_dir}/sources.csv", index=False)
        
        print(f"\n🎉 PROCESAMIENTO COMPLETADO")
        print(f"📊 Total chunks: {len(all_chunks)}")
        print(f"📚 Documentos procesados: {len(sources)}")
        print(f"📄 PDFs: {len([s for s in sources if s['formato'] == 'pdf'])}")
        print(f"📝 TXT/MD: {len([s for s in sources if s['formato'] == 'txt'])}")
        
    else:
        print("❌ No se pudieron procesar chunks de ningún documento")
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