from datetime import datetime
import time
from html import parser
import os
import json
import argparse
import logging

from typing import Optional, List, Dict
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.prompts import generate_system_prompt
from eval.evaluate import evaluate_response

from metrics_logger import metrics_logger

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
        .metrics-button { 
            background: #3498db; 
            color: white; 
            padding: 8px 15px; 
            text-decoration: none; 
            border-radius: 5px; 
            font-size: 14px;
            float: right;
            margin-top: -50px;
        }
        .metrics-button:hover {
            background: #2980b9;
        }
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-container">
            <h1>UFRO ChatBot</h1>
            <a href="/metrics" class="metrics-button">📊 Ver Métricas</a>
        </div>
        
        <div class="input-group">
            <input type="text" id="query" placeholder="Ingrese su consulta sobre normativa UFRO..." onkeypress="handleKeyPress(event)">
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
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuery();
            }
        }
        
        async function sendQuery() {
            const query = document.getElementById('query').value;
            const provider = document.getElementById('provider').value;
            const loading = document.getElementById('loading');
            const responseDiv = document.getElementById('response');
            
            if (!query) {
                alert('Por favor, ingrese una consulta');
                return;
            }
            
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
                    // Formatear la respuesta con mejor presentación
                    let citationsHtml = '';
                    if (data.citations && data.citations.length > 0) {
                        citationsHtml = `<div class="citations">
                            <strong>📚 Referencias:</strong><br>
                            ${data.citations.join('<br>')}
                        </div>`;
                    }
                    
                    responseDiv.innerHTML = `
                        <div class="response">
                            <strong>🤖 Respuesta:</strong><br>
                            ${data.answer.replace(/\\n/g, '<br>')}
                            ${citationsHtml}
                            <div style="font-size: 12px; color: #888; margin-top: 10px;">
                                ⏱️ Latencia: ${data.latency}s | 🚀 Proveedor: ${data.provider}
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                responseDiv.innerHTML = '<div class="response">❌ Error de conexión con el servidor</div>';
            } finally {
                loading.style.display = 'none';
                // Limpiar el campo de entrada después de una consulta exitosa
                document.getElementById('query').value = '';
            }
        }
        
        // Enfocar el campo de entrada al cargar la página
        window.onload = function() {
            document.getElementById('query').focus();
        }
    </script>
</body>
</html>
'''


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/metrics')
def show_metrics():
    """Muestra métricas en interfaz web"""
    html_table = metrics_logger.generate_comparative_table()
    return render_template_string(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Métricas - UFRO Assistant</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
            .back-button {{ background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Métricas del Sistema</h1>
            <a href="/" class="back-button">← Volver al Chat</a>
        </div>
        {html_table}

  
    <div style="text-align: right; margin-bottom: 20px;">
        <a href="/metrics" style="background: #3498db; color: white; padding: 8px 15px; text-decoration: none; border-radius: 5px; font-size: 14px;">
            📊 Ver Métricas
        </a>
    </div>
    </body>
    </html>
    """)

@app.route('/api/metrics/recent')
def api_recent_metrics():
    """API para obtener métricas recientes"""
    recent = metrics_logger.get_recent_queries(10)
    return jsonify(recent)

@app.route('/api/metrics/stats')
def api_provider_stats():
    """API para obtener estadísticas de proveedores"""
    stats = metrics_logger.get_provider_stats()
    return jsonify(stats)



class UFROAssistant:
    def __init__(self):
        self.providers = {}
        self.retriever = None
        self.generate_system_prompt = None
        
        try:
            # Importar dentro del try para manejar errores
            from providers.chatgpt import ChatGPTProvider
            from providers.deepseek import DeepSeekProvider
            from rag.retrieve import Retriever
            from rag.prompts import generate_system_prompt
            
            # Inicializar proveedores
            self.providers = {
                'chatgpt': ChatGPTProvider(),
                'deepseek': DeepSeekProvider()
            }
            
            # Inicializar retriever
            self.retriever = Retriever()
            
            # Guardar referencia a la función
            self.generate_system_prompt = generate_system_prompt
            
            print("✅ UFRO Assistant inicializado correctamente")
            print(f"✅ Proveedores cargados: {list(self.providers.keys())}")
            
        except ImportError as e:
            print(f"❌ Error de importación: {e}")
            print("⚠️  Algunos componentes no están disponibles")
        except Exception as e:
            print(f"❌ Error inicializando UFRO Assistant: {e}")
            # Mantener el sistema funcional con valores por defecto
            self.providers = {}
            self.retriever = None
    
    def is_initialized(self):
        """Verifica si el assistant está correctamente inicializado"""
        return (self.retriever is not None and 
                len(self.providers) > 0 and 
                self.generate_system_prompt is not None)
    
    def process_query(self, query: str, provider_name: str = 'chatgpt', k: int = 4) -> Dict:
        # ✅ Usar import time global
        import time as time_module
        start_time = time_module.time()
        retrieval_start = time_module.time()
        
        # Verificar inicialización
        if not self.is_initialized():
            return {
                'answer': '⚠️ El sistema no está completamente inicializado. Algunos componentes pueden no estar disponibles.',
                'citations': [],
                'latency': 0,
                'provider': 'none',
                'retrieval_latency': 0,
                'llm_latency': 0,
                'documents_retrieved': 0
            }
        
        # 1. Recuperar documentos relevantes
        retrieved_docs = self.retriever.search(query, k=k)
        retrieval_latency = time_module.time() - retrieval_start
        
        if not retrieved_docs:
            result = {
                'answer': 'No encontré información relevante en la normativa UFRO sobre este tema.',
                'citations': [],
                'latency': time_module.time() - start_time,
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': 0,
                'documents_retrieved': 0
            }
            
            # Loggear métricas
            self._log_metrics(query, result)
            return result
        
        # 2. Generar contexto
        context = "\n\n".join([
            f"[Documento: {doc.get('title', 'Unknown')}, Página: {doc.get('page', 1)}]\n{doc.get('content', '')}"
            for doc in retrieved_docs
        ])
        
        # 3. Generar respuesta con el proveedor
        try:
            llm_start = time_module.time()
            
            provider = self.providers.get(provider_name)
            if not provider:
                # Fallback al primer proveedor disponible
                available_providers = list(self.providers.keys())
                if available_providers:
                    provider_name = available_providers[0]
                    provider = self.providers[provider_name]
                    print(f"⚠️  Proveedor solicitado no disponible. Usando {provider_name} como fallback.")
                else:
                    result = {
                        'answer': '❌ No hay proveedores LLM disponibles en este momento.',
                        'citations': [],
                        'latency': time_module.time() - start_time,
                        'provider': 'none',
                        'retrieval_latency': round(retrieval_latency, 2),
                        'llm_latency': 0,
                        'documents_retrieved': len(retrieved_docs)
                    }
                    self._log_metrics(query, result)
                    return result
            
            # Usar la función de prompt del sistema
            system_prompt = self.generate_system_prompt()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
            ]
            
            response = provider.chat(messages)
            llm_latency = time_module.time() - llm_start
            
            # 4. Extraer citas
            citations = []
            for doc in retrieved_docs:
                citations.append(f"[{doc.get('title', 'Unknown')}, p.{doc.get('page', 1)}]")
            
            # 5. Crear resultado final
            total_latency = time_module.time() - start_time
            result = {
                'answer': response,
                'citations': citations[:3],
                'latency': round(total_latency, 2),
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': round(llm_latency, 2),
                'documents_retrieved': len(retrieved_docs)
            }
            
            # 6. Loggear métricas
            self._log_metrics(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando query: {e}")
            result = {
                'answer': f'❌ Error procesando la consulta: {str(e)}',
                'citations': [],
                'latency': round(time_module.time() - start_time, 2),
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': 0,
                'documents_retrieved': len(retrieved_docs)
            }
            
            # Loggear incluso los errores
            self._log_metrics(query, result)
            
            return result
    
    def _log_metrics(self, query: str, result: Dict):
        """Método helper para loggear métricas"""
        try:
            from metrics_logger import metrics_logger
            metrics_data = {
                'query': query,
                'provider': result['provider'],
                'answer': result['answer'],
                'latency': result['latency'],
                'retrieval_latency': result.get('retrieval_latency', 0),
                'llm_latency': result.get('llm_latency', 0),
                'citations': result['citations'],
                'documents_retrieved': result.get('documents_retrieved', 0)
            }
            metrics_logger.log_query(metrics_data)
        except Exception as e:
            print(f"⚠️ Error loggeando métricas: {e}")
    def _extract_citations(self, text: str) -> List[str]:
        """Extrae citas del formato [Documento, página X]"""
        import re
        citations = re.findall(r'\[([^,\]]+),\s*página?\s*(\d+)\]', text, re.IGNORECASE)
        return [f"{doc}, p.{page}" for doc, page in citations]

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
    """Modo CLI mejorado con UX profesional"""
    parser = argparse.ArgumentParser(description='🤖 UFRO Assistant - Sistema RAG de Normativa')
    parser.add_argument('query', nargs='?', help='Consulta sobre normativa UFRO')
    parser.add_argument('--provider', choices=['chatgpt', 'deepseek', 'auto'], default='deepseek', 
                       help='Proveedor LLM (auto=selección automática)')
    parser.add_argument('--k', type=int, default=4, help='Número de documentos a recuperar (default: 4)')
    parser.add_argument('--batch', help='Archivo JSONL con preguntas para evaluación batch')
    parser.add_argument('--web', action='store_true', help='Iniciar servidor web Flask')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas del sistema')
    
    args = parser.parse_args()
    
    if args.stats:
        show_system_stats()
        return
    
    if args.web:
        start_web_server()
        return
    
    if args.batch:
        run_batch_evaluation(args)
        return
    
    if args.query:
        run_single_query(args)
    else:
        run_interactive_mode(args)


def show_system_stats():
    """Muestra estadísticas del sistema"""
    print("📊 ESTADÍSTICAS DEL SISTEMA UFRO ASSISTANT")
    print("=" * 50)
    
    try:
        from rag.retrieve import Retriever
        retriever = Retriever()
        stats = retriever.get_retrieval_stats()
        print("Índice FAISS:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")

def start_web_server():
    """Inicia el servidor web Flask"""
    print("🚀 Iniciando servidor Flask en http://0.0.0.0:5000")
    print("📱 Accede a la interfaz web: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
def run_interactive_mode(args):
    """Modo interactivo"""
    print("🤖 UFRO Assistant - Modo interactivo")
    print("Escriba 'salir' para terminar\n")
    
    while True:
        try:
            query = input("Consulta: ").strip()
            if query.lower() in ['salir', 'exit', 'quit']:
                break
            
            if query:
                result = assistant.process_query(query, args.provider, args.k)
                print(f"\nRespuesta ({result['provider']}):")
                print(result['answer'])
                if result['citations']:
                    print(f"\nReferencias:")
                    for cite in result['citations']:
                        print(f"  • {cite}")
                print(f"\n[Tiempo: {result['latency']}s]")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_single_query(args):
    """Ejecuta una consulta única con formato profesional"""
    from rag.retrieve import Retriever
    from providers.router import ProviderRouter
    
    print("🔍 UFRO Assistant - Procesando consulta...")
    print(f"📝 Consulta: {args.query}")
    print(f"🤖 Proveedor: {args.provider}")
    print(f"📚 Documentos a recuperar: {args.k}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        retriever = Retriever()
        router = ProviderRouter()
        
        # Recuperar documentos
        docs = retriever.search(args.query, args.k)
        retrieval_time = time.time() - start_time
        
        print(f"✅ Recuperados {len(docs)} documentos ({retrieval_time:.2f}s)")
        
        # Generar respuesta
        provider_to_use = None if args.provider == 'auto' else args.provider
        context = "\n".join([f"- {doc['title']} (p.{doc['page']}): {doc['content'][:200]}..." for doc in docs])
        
        messages = [
            {"role": "system", "content": generate_system_prompt()},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {args.query}"}
        ]
        
        response, provider, llm_time = router.chat(messages, provider_to_use)
        total_time = time.time() - start_time
        
        # Mostrar resultados
        print(f"\n🤖 RESPUESTA ({provider.upper()}):")
        print("─" * 60)
        print(response)
        print("─" * 60)
        
        print(f"\n📚 REFERENCIAS:")
        for doc in docs[:3]:  # Máximo 3 referencias
            print(f"  • {doc['title']} - Página {doc['page']} (score: {doc['score']:.3f})")
        
        print(f"\n⏱️  MÉTRICAS:")
        print(f"  • Tiempo total: {total_time:.2f}s")
        print(f"  • Recuperación: {retrieval_time:.2f}s") 
        print(f"  • Generación: {llm_time:.2f}s")
        print(f"  • Documentos recuperados: {len(docs)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def run_batch_evaluation(args):
    """Ejecuta evaluación batch con manejo de errores mejorado"""
    print("📊 EJECUTANDO EVALUACIÓN BATCH...")
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(args.batch):
            print(f"❌ Archivo no encontrado: {args.batch}")
            return
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [json.loads(line) for line in f if line.strip()]
        
        if not questions:
            print("❌ No hay preguntas en el archivo batch")
            return
        
        print(f"🔍 Procesando {len(questions)} preguntas...")
        
        results = []
        for i, q in enumerate(questions, 1):
            print(f"  {i}/{len(questions)}: {q.get('question', '')[:50]}...")
            
            try:
                result = assistant.process_query(q.get('question', ''), args.provider, args.k)
                
                # Evaluar calidad con manejo de errores
                try:
                    from eval.evaluate import Evaluator
                    evaluator = Evaluator()
                    metrics = evaluator.evaluate_response(result, q)
                except Exception as e:
                    print(f"    ⚠️  Error en evaluación: {e}")
                    # Métricas por defecto si falla la evaluación
                    metrics = {
                        'exact_match': 0.0,
                        'semantic_similarity': 0.0,
                        'citation_coverage': 1.0 if result['citations'] else 0.0,
                        'has_answer': len(result['answer'].strip()) > 0,
                        'answer_length': len(result['answer'])
                    }
                
                results.append({
                    'question': q.get('question', ''),
                    'expected_answer': q.get('expected_answer', ''),
                    'actual_answer': result['answer'],
                    'provider': result['provider'],
                    'citations': ', '.join(result['citations']),
                    'latency': result['latency'],
                    'retrieval_latency': result.get('retrieval_latency', 0),
                    'llm_latency': result.get('llm_latency', 0),
                    'exact_match': metrics.get('exact_match', 0),
                    'semantic_similarity': metrics.get('semantic_similarity', 0),
                    'citation_coverage': metrics.get('citation_coverage', 0),
                    'has_answer': metrics.get('has_answer', False),
                    'answer_length': metrics.get('answer_length', 0)
                })
                
            except Exception as e:
                print(f"    ❌ Error procesando pregunta {i}: {e}")
                # Registrar error pero continuar
                results.append({
                    'question': q.get('question', ''),
                    'expected_answer': q.get('expected_answer', ''),
                    'actual_answer': f'ERROR: {str(e)}',
                    'provider': 'error',
                    'citations': '',
                    'latency': 0,
                    'retrieval_latency': 0,
                    'llm_latency': 0,
                    'exact_match': 0,
                    'semantic_similarity': 0,
                    'citation_coverage': 0,
                    'has_answer': False,
                    'answer_length': 0
                })
        
        # Guardar CSV
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Generar reporte
        generate_evaluation_report(df, filename)
        
        print(f"✅ Evaluación completada: {filename}")
        
    except Exception as e:
        print(f"❌ Error en evaluación batch: {e}")
    
    print(f"✅ Evaluación completada: {filename}")
def show_metrics_cli():
        """Muestra métricas en CLI"""
        print("📊 MÉTRICAS DEL SISTEMA UFRO ASSISTANT")
        print("=" * 60)
        
        # Generar y mostrar tabla comparativa
        html_table = metrics_logger.generate_comparative_table()
        
        # Extraer solo el contenido de la tabla para CLI
        try:
            import re
            # Extraer datos de la tabla HTML
            table_data = re.findall(r'<td>(.*?)</td>', html_table)
            
            if table_data:
                providers = set()
                current_provider = None
                print("\n" + "─" * 100)
                print(f"{'Proveedor':<12} {'Consultas':<10} {'Latencia (s)':<12} {'Costo Avg':<12} {'Costo Total':<12} {'Tokens':<8} {'Citas':<8}")
                print("─" * 100)
                
                # Leer datos del archivo CSV directamente
                stats_file = Path("logs/reportes/provider_stats.csv")
                if stats_file.exists():
                    df = pd.read_csv(stats_file)
                    for _, row in df.iterrows():
                        print(f"{row['provider']:<12} {row['total_queries']:<10} {row['avg_latency']:<12.2f} "
                            f"${row['avg_cost']:<11.6f} ${row['total_cost']:<11.6f} {row['avg_tokens']:<8.0f} {row.get('success_rate', 0):<8.1f}")
                
                print("─" * 100)
                
            # Mostrar consultas recientes
            recent = metrics_logger.get_recent_queries(5)
            if recent:
                print(f"\n🕒 ÚLTIMAS 5 CONSULTAS:")
                for i, query in enumerate(recent, 1):
                    print(f"  {i}. [{query['provider']}] {query['query'][:50]}... "
                        f"(Latencia: {query['latency_total']:.2f}s, Costo: ${query['estimated_cost_usd']:.6f})")
                    
        except Exception as e:
            print(f"❌ Error mostrando métricas: {e}")

        parser.add_argument('--metrics', action='store_true', help='Mostrar métricas del sistema')

def generate_evaluation_report(df, filename):
    """Genera reporte de evaluación ejecutivo"""
    report = f"""
📊 INFORME DE EVALUACIÓN - UFRO ASSISTANT
==========================================
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Muestras: {len(df)}
Archivo: {filename}

MÉTRICAS POR PROVEEDOR:
"""
    
    for provider in df['provider'].unique():
        provider_data = df[df['provider'] == provider]
        report += f"""
🤖 {provider.upper()}:
  • Exact Match: {provider_data['exact_match'].mean():.3f}
  • Similitud Semántica: {provider_data['semantic_similarity'].mean():.3f}
  • Cobertura de Citas: {provider_data['citation_coverage'].mean():.3f}
  • Latencia Promedio: {provider_data['latency'].mean():.2f}s
  • Muestras: {len(provider_data)}
"""
    
    # Hallazgos clave
    report += """
🔍 HALLAZGOS PRINCIPALES:
"""
    
    if len(df['provider'].unique()) > 1:
        best_provider = df.groupby('provider')['semantic_similarity'].mean().idxmax()
        fastest_provider = df.groupby('provider')['latency'].mean().idxmin()
        
        report += f"  1. Mejor calidad: {best_provider.upper()}\n"
        report += f"  2. Más rápido: {fastest_provider.upper()}\n"
    
    report += f"  3. Cobertura de citas: {df['citation_coverage'].mean()*100:.1f}%\n"
    report += f"  4. Tiempo promedio: {df['latency'].mean():.2f}s\n"
    
    # Guardar reporte
    report_filename = filename.replace('.csv', '_report.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"📄 Reporte guardado: {report_filename}")

if __name__ == '__main__':
    cli_mode()
