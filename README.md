# 🚀 UFRO Normativa Assistant - Instrucciones de Despliegue

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

ufro-assistant/
├── app.py                 # ✅ CLI + Servidor web
├── providers/            # ✅ ChatGPT + DeepSeek + Router
├── rag/                  # ✅ Ingesta + Embeddings + FAISS
├── eval/                 # ✅ Métricas completas + Reportes
├── data/                 # ✅ Documentos + Índices
├── ETHICS.md            # ✅ Políticas éticas
└── scripts/             # ✅ Scripts de automatización