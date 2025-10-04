import os
import json
import argparse
import logging
from typing import Optional, List, Dict
from flask import Flask, request, jsonify, render_template_string
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.prompts import generate_system_prompt
from eval.evaluate import evaluate_response

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# HTML Template para la interfaz web
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UFRO Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
        h1 { color: #333; }
        .input-group { margin: 20px 0; }
        input, select, button { padding: 10px; margin: 5px; font-size: 16px; }
        input { width: 60%; }
        button { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .response { background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .citations { font-size: 14px; color: #666; margin-top: 10px; }
        .loading { display: none; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>UFRO Normativa Assistant</h1>
        <div class="input-group">
            <input type="text" id="query" placeholder="Ingrese su consulta sobre normativa UFRO...">
            <select id="provider">
                <option value="chatgpt">ChatGPT</option>
                <option value="deepseek">DeepSeek</option>
            </select>
            <button onclick="sendQuery()">Consultar</button>
        </div>
        <div class="loading" id="loading">Procesando consulta...</div>
        <div id="response"></div>
    </div>
    
    <script>
        async function sendQuery() {
            const query = document.getElementById('query').value;
            const provider = document.getElementById('provider').value;
            const loading = document.getElementById('loading');
            const responseDiv = document.getElementById('response');
            
            if (!query) return;
            
            loading.style.display = 'block';
            responseDiv.innerHTML = '';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, provider})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    responseDiv.innerHTML = '<div class="response">Error: ' + data.error + '</div>';
                } else {
                    responseDiv.innerHTML = `
                        <div class="response">
                            <strong>Respuesta:</strong><br>
                            ${data.answer}
                            <div class="citations">
                                <strong>Referencias:</strong><br>
                                ${data.citations.join('<br>')}
                            </div>
                            <div style="font-size: 12px; color: #888; margin-top: 10px;">
                                Latencia: ${data.latency}s | Proveedor: ${data.provider}
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                responseDiv.innerHTML = '<div class="response">Error de conexión</div>';
            } finally {
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''


class UFROAssistant:
    def __init__(self):
        self.providers = {
            'chatgpt': ChatGPTProvider(),
            'deepseek': DeepSeekProvider()
        }
        self.retriever = Retriever()
        
    def process_query(self, query: str, provider_name: str = 'chatgpt', k: int = 4) -> Dict:
        """Procesa una consulta usando RAG"""
        import time
        start_time = time.time()
        
        # Recuperar documentos relevantes
        retrieved_docs = self.retriever.search(query, k=k)
        
        if not retrieved_docs:
            return {
                'answer': 'No encontré información relevante en la normativa UFRO. '
                         'Le sugiero contactar a la oficina académica correspondiente.',
                'citations': [],
                'latency': time.time() - start_time,
                'provider': provider_name
            }
        
        # Generar contexto
        context = "\n\n".join([
            f"[Documento: {doc['title']}, Página: {doc['page']}]\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Generar respuesta con el proveedor seleccionado
        provider = self.providers[provider_name]
        system_prompt = generate_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
        ]
        
        response = provider.chat(messages)
        
        # Extraer citas
        citations = []
        for doc in retrieved_docs:
            citations.append(f"[{doc['title']}, p.{doc['page']}]")
        
        return {
            'answer': response,
            'citations': citations[:3],  # Limitar a 3 citas
            'latency': round(time.time() - start_time, 2),
            'provider': provider_name
        }

assistant = UFROAssistant()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        data = request.json
        query = data.get('query', '')
        provider = data.get('provider', 'chatgpt')
        
        if not query:
            return jsonify({'error': 'Query vacía'}), 400
        
        result = assistant.process_query(query, provider)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error procesando query: {e}")
        return jsonify({'error': str(e)}), 500

def cli_mode():
    """Modo CLI para uso local"""
    parser = argparse.ArgumentParser(description='UFRO Assistant CLI')
    parser.add_argument('query', nargs='?', help='Consulta sobre normativa UFRO')
    parser.add_argument('--provider', choices=['chatgpt', 'deepseek'], default='chatgpt')
    parser.add_argument('--k', type=int, default=4, help='Número de documentos a recuperar')
    parser.add_argument('--batch', help='Archivo JSON con preguntas para evaluar')
    parser.add_argument('--web', action='store_true', help='Iniciar servidor web')
    
    args = parser.parse_args()
    
    if args.web:
        print("Iniciando servidor Flask en http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        return
    
    if args.batch:
        # Modo batch
        with open(args.batch, 'r') as f:
            questions = [json.loads(line) for line in f]
        
        results = []
        for q in questions:
            result = assistant.process_query(q['question'], args.provider, args.k)
            results.append({
                'question': q['question'],
                'provider': args.provider,
                'answer': result['answer'],
                'references': ', '.join(result['citations']),
                'latency': result['latency']
            })
        
        # Guardar resultados
        import csv
        with open('batch_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'provider', 'answer', 'references', 'latency'])
            writer.writeheader()
            writer.writerows(results)
        print("Resultados guardados en batch_results.csv")
        
    elif args.query:
        # Modo consulta única
        result = assistant.process_query(args.query, args.provider, args.k)
        print(f"\nRespuesta ({args.provider}):")
        print(result['answer'])
        print(f"\nReferencias:")
        for cite in result['citations']:
            print(f"  {cite}")
        print(f"\nLatencia: {result['latency']}s")
    else:
        # Modo interactivo
        print("UFRO Assistant - Modo interactivo")
        print("Escriba 'salir' para terminar\n")
        
        while True:
            query = input("Consulta: ").strip()
            if query.lower() == 'salir':
                break
            
            if query:
                result = assistant.process_query(query, args.provider, args.k)
                print(f"\nRespuesta:")
                print(result['answer'])
                print(f"\nReferencias:")
                for cite in result['citations']:
                    print(f"  {cite}")
                print()

if __name__ == '__main__':
    cli_mode()
