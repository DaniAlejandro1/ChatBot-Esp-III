FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependencias del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        pkg-config \
        curl \
        libopenblas-dev \
        liblapack-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements y código
COPY requirements.txt /app/requirements.txt
COPY . /app

# Instala dependencias de Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install pyarrow pandas

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

# CMD final: ejecuta ingest y embed al iniciar el contenedor
CMD ["sh", "-c", "python -m rag.ingest && python -m rag.embed && python app.py"]
