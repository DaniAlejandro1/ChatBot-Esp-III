# 🎓 UFRO ChatBot
Sistema RAG (Retrieval-Augmented Generation) especializado en normativa y reglamentos de la Universidad de La Frontera. Proporciona respuestas precisas con citas verificables utilizando múltiples proveedores de IA.

## ⚡ Instalación Rápida (5 minutos)

```bash
# 1. Clonar y configurar
git clone https://github.com/tu-usuario/ufro-assistant.git
cd ufro-assistant
cp .env.example .env

# 2. Instalar dependencias
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configurar API keys (EDITAR .env)
echo "OPENAI_API_KEY=tu_key_aqui" >> .env
echo "DEEPSEEK_API_KEY=tu_key_aqui" >> .env

# 4. Procesar documentos
python -m rag.ingest
python -m rag.embed

# 5. ¡Listo! Probar el sistema
python app.py "¿Cuáles son los requisitos de titulación?"


# Evaluación batch completa
python app.py --batch eval/gold_set.jsonl --provider auto

# Evaluación individual  
python app.py "¿Cuál es el porcentaje de asistencia requerido?" --provider chatgpt

# Servidor web
python app.py --web

#Estructura de carpetas
ufro-assistant/
├── app.py                 # ✅ CLI + Servidor web
├── providers/            # ✅ ChatGPT + DeepSeek + Router
├── rag/                  # ✅ Ingesta + Embeddings + FAISS
├── eval/                 # ✅ Métricas completas + Reportes
├── data/                 # ✅ Documentos + Índices
├── logs/                 # ✅ Loggin de procesos
├── ETHICS.md            # ✅ Políticas éticas
└── scripts/             # ✅ Scripts de automatización
```
# Herramientas de Diagnóstico

```bash
# Ver métricas del sistema
    python app.py --metrics

# Estadísticas técnicas

    python app.py --stats

# Verificar estado de componentes

python diagnose_system.py
```
# 📊 Evaluación por Lotes (Batch)

## Preparación de Datos
```bash
# Estructura del archivo batch (JSONL)

echo '{"question": "¿Cuál es la asistencia mínima?", "expected_answer": "75% según reglamento", "category": "asistencia"}' >> eval/mi_evaluacion.jsonl
echo '{"question": "¿Plazos de matrícula?", "expected_answer": "Enero y julio", "category": "calendario"}' >> eval/mi_evaluacion.jsonl
```
## Ejecución de Evaluación

```bash

# Evaluación simple

python app.py --batch eval/gold_set.jsonl --provider chatgpt


# Evaluación comparativa`

python app.py --batch eval/gold_set.jsonl --provider chatgpt
python app.py --batch eval/gold_set.jsonl --provider deepseek

# Generar reporte comparativo
python scripts/generate_comparison.py
```

## Archivos de Resultados
``batch_results_TIMESTAMP.csv`` - Resultados detallados

``evaluation_report.txt`` - Análisis ejecutivo

``logs/reportes/`` - Métricas y tablas comparativas


## Componentes del Pipeline

1) 📄 Ingesta de Documentos (rag/ingest.py)

    Extracción de texto desde PDF

    División en chunks semánticos

    Metadatos: documento, página, vigencia

2) 📊 Vectorización (rag/embed.py)

    Generación de embeddings con Sentence Transformers

    Índice FAISS para búsqueda rápida

    Normalización para similitud coseno

3. 🔍 Recuperación (rag/retrieve.py)

    Búsqueda semántica de chunks relevantes

    Ranking por similitud

    Filtrado por umbral de relevancia

4. 🤖 Generación (providers/)

    Integración multi-proveedor

    Prompt engineering especializado

    Formato consistente de respuestas

5. 📈 Evaluación (eval/)

    Métricas de calidad automáticas

    Comparativa entre proveedores

    Logging de métricas operativas

6. ⚖️ Políticas del Sistema
    Política de Abstención
    El sistema está programado para abstenerse de responder cuando:

```python
# Casos de abstención automática
CONDICIONES_ABSTENCION = [
    "No se encuentran documentos relevantes en la base de conocimiento",
    "La consulta requiere interpretación legal o jurídica", 
    "Información personalizada sobre casos específicos",
    "Temas fuera del ámbito normativo UFRO",
    "Documentos con vigencia expirada"
]
# Mensaje estándar de abstención
"No encontré información específica sobre este tema en la normativa UFRO disponible. Le recomiendo contactar con la oficina académica correspondiente para asistencia personalizada."
```


# Política de Privacidad
#### 🔒 Datos que NO se Almacenan
- Información personal de usuarios

- Consultas identificables

- Historial de conversaciones personales

- Datos de contacto o identificación

#### 📊 Datos que SI se Almacenan
- Métricas agregadas de rendimiento

- Estadísticas de uso anónimas

- Logs de errores técnicos

- Métricas de calidad de respuestas

#### ⏰ Retención de Datos
- Logs de métricas: 30 días

- Archivos de evaluación: 90 días

- Documentos procesados: Hasta nueva versión

- Índices de búsqueda: Actualizados con documentos

- Estructura de Metadatos por Chunk
```json
{
  "chunk_id": "reglamento_academico_p12_c3",
  "doc_id": "reglamento_academico.pdf",
  "title": "Reglamento Académico General",
  "page": 12,
  "content": "Texto del chunk...",
  "source": "data/raw/reglamento_academico.pdf",
  "vigencia": "2024-2025",
  "url_oficial": "https://www.ufro.cl/normativa/academico",
  "actualizacion": "2024-01-15"
}
```

# Soporte Técnico
## Para asistencia técnica:

📋 Verificar logs en logs/reportes/

🔍 Ejecutar diagnóstico: python diagnose_system.py

📝 Reportar issue con métricas relevantes
