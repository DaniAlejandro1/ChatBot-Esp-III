from datetime import datetime, timezone  # CAMBIADO: agregar timezone
import time
import os
import json
import argparse
import logging
import uuid
from typing import Optional, List, Dict
from flask import Flask, request, jsonify
import pandas as pd
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
import rag.ingest as ingest
import rag.embed as embed
from rag.retrieve import Retriever
from rag.prompts import generate_system_prompt
from eval.evaluate import evaluate_response

from metrics_logger import metrics_logger
from mongo_logger import mongo_logger

# ==================== FUNCIONES AUXILIARES PP3 ====================

def format_citations_pp3(citations):
    """Convertir citas al formato PP3: {"doc": "...", "page": "..."}"""
    formatted = []
    for cite in citations:
        # Ejemplo: "[Reglamento, p.40]" -> {"doc": "Reglamento", "page": "40"}
        if cite.startswith('[') and cite.endswith(']'):
            content = cite[1:-1]  # Quitar corchetes
            if ', p.' in content:
                parts = content.split(', p.')
                if len(parts) == 2:
                    formatted.append({
                        "doc": parts[0].strip(),
                        "page": parts[1].strip()
                    })
                else:
                    formatted.append({"doc": content, "page": "1"})
            elif ', pág.' in content:
                parts = content.split(', pág.')
                if len(parts) == 2:
                    formatted.append({
                        "doc": parts[0].strip(),
                        "page": parts[1].strip()
                    })
                else:
                    formatted.append({"doc": content, "page": "1"})
            else:
                formatted.append({"doc": content, "page": "1"})
        else:
            formatted.append({"doc": cite, "page": "1"})
    return formatted

def format_response_pp3(result):
    """Formatear respuesta según especificación PP3"""
    return {
        "decision": "success" if result.get('success', False) else "error",
        "normativa_answer": {
            "text": result.get('answer', ''),
            "citations": format_citations_pp3(result.get('citations', []))
        },
        "timing_ms": round(result.get('latency', 0) * 1000, 1),
        "request_id": result.get('request_id', ''),
        "metadata": {
            "provider": result.get('provider', 'unknown'),
            "retrieval_latency_ms": round(result.get('retrieval_latency', 0) * 1000, 1),
            "llm_latency_ms": round(result.get('llm_latency', 0) * 1000, 1),
            "documents_retrieved": result.get('documents_retrieved', 0)
        }
    }

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ==================== ENDPOINTS DE MONGO METRICS (API JSON) ====================

@app.route('/api/mongo/status', methods=['GET'])
def api_mongo_status():
    """API para verificar estado de MongoDB"""
    return jsonify({
        'connected': mongo_logger.is_connected(),
        'database': mongo_logger.db.name if mongo_logger.db else None,
        'timestamp': datetime.now(timezone.utc).isoformat(),  # CAMBIADO
        'service': 'pp1_chatbot'
    })

@app.route('/api/mongo/metrics/summary', methods=['GET'])
def api_mongo_summary():
    """API para obtener resumen de métricas desde MongoDB"""
    days = request.args.get('days', default=7, type=int)
    
    if not mongo_logger.is_connected():
        return jsonify({
            'error': 'MongoDB no conectado',
            'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
        }), 503
    
    metrics = mongo_logger.get_metrics_summary(days)
    
    return jsonify({
        'summary': metrics,
        'period_days': days,
        'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
    })

@app.route('/api/mongo/metrics/recent', methods=['GET'])
def api_mongo_recent():
    """API para obtener logs recientes desde MongoDB"""
    limit = request.args.get('limit', default=20, type=int)
    
    if not mongo_logger.is_connected():
        return jsonify({
            'error': 'MongoDB no conectado',
            'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
        }), 503
    
    logs = mongo_logger.get_recent_logs(limit)
    
    # Convertir ObjectId a string para JSON serialization
    for log in logs:
        log['_id'] = str(log['_id'])
        if 'timestamp' in log and hasattr(log['timestamp'], 'isoformat'):
            log['timestamp'] = log['timestamp'].isoformat()
    
    return jsonify({
        'logs': logs,
        'count': len(logs),
        'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
    })

# ==================== ENDPOINTS DE MÉTRICAS LOCALES ====================

@app.route('/api/metrics/recent', methods=['GET'])
def api_recent_metrics():
    """API para obtener métricas recientes locales"""
    limit = request.args.get('limit', default=10, type=int)
    recent = metrics_logger.get_recent_queries(limit)
    return jsonify({
        'recent_queries': recent,
        'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
    })

@app.route('/api/metrics/stats', methods=['GET'])
def api_provider_stats():
    """API para obtener estadísticas de proveedores"""
    stats = metrics_logger.get_provider_stats()
    return jsonify({
        'provider_stats': stats,
        'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
    })

# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint para n8n"""
    mongo_status = mongo_logger.is_connected()
    
    health_status = {
        'status': 'healthy',
        'service': 'pp1_chatbot',
        'timestamp': datetime.now(timezone.utc).isoformat(),  # CAMBIADO
        'components': {
            'mongo': 'connected' if mongo_status else 'disconnected',
            'chatbot': 'ready'
        },
        'checks': {
            'mongo_connection': mongo_status,
            'chatbot_initialized': assistant.is_initialized()
        }
    }
    
    # Si MongoDB no está conectado, cambiar status a degraded pero no unhealthy
    if not mongo_status:
        health_status['status'] = 'degraded'
        health_status['warning'] = 'MongoDB no conectado, logging degradado'
    
    if not assistant.is_initialized():
        health_status['status'] = 'unhealthy'
        health_status['error'] = 'Chatbot no inicializado correctamente'
    
    return jsonify(health_status)

# ==================== UFRO ASSISTANT CLASS ====================

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
            
            logger.info("✅ UFRO Assistant inicializado correctamente")
            logger.info(f"✅ Proveedores cargados: {list(self.providers.keys())}")
            
            # Verificar conexión MongoDB
            if mongo_logger.is_connected():
                logger.info("✅ MongoDB conectado para logging")
            else:
                logger.warning("⚠️  MongoDB no disponible, usando logging local")
            
        except ImportError as e:
            logger.error(f"❌ Error de importación: {e}")
            logger.error("⚠️  Algunos componentes no están disponibles")
        except Exception as e:
            logger.error(f"❌ Error inicializando UFRO Assistant: {e}")
            # Mantener el sistema funcional con valores por defecto
            self.providers = {}
            self.retriever = None
    
    def is_initialized(self):
        """Verifica si el assistant está correctamente inicializado"""
        return (self.retriever is not None and 
                len(self.providers) > 0 and 
                self.generate_system_prompt is not None)
    
    def process_query(self, query: str, provider_name: str = 'chatgpt', k: int = 4, 
                      user_id: str = None, user_type: str = None) -> Dict:
        """Procesa una consulta y retorna respuesta con métricas"""
        
        start_time = time.time()
        retrieval_start = time.time()
        
        # Generar request_id único
        request_id = str(uuid.uuid4())
        
        # Preparar datos para logging en MongoDB
        access_log_data = {
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc),  # CAMBIADO
            'route': '/api/query',
            'user': {
                'id': user_id or 'anonymous',
                'type': user_type or 'unknown',
                'role': 'basic'
            },
            'input': {
                'has_question': True,
                'question': query[:500],  # Limitar tamaño
                'question_length': len(query),
                'provider': provider_name,
                'k_documents': k
            },
            'service_type': 'pp1_chatbot',
            'endpoint': f'/api/query?provider={provider_name}'
        }
        
        # Verificar inicialización
        if not self.is_initialized():
            result = {
                'answer': '⚠️ El sistema no está completamente inicializado. Algunos componentes pueden no estar disponibles.',
                'citations': [],
                'latency': 0,
                'provider': 'none',
                'retrieval_latency': 0,
                'llm_latency': 0,
                'documents_retrieved': 0,
                'request_id': request_id,
                'success': False,
                'error': 'system_not_initialized'
            }
            
            # Loggear métricas locales
            self._log_metrics(query, result)
            
            # Loggear en MongoDB (error)
            access_log_data.update({
                'decision': 'error',
                'timing_ms': 0,
                'status_code': 500,
                'errors': 'Sistema no inicializado',
                'pp1_used': True,
                'metadata': {
                    'error_type': 'initialization',
                    'retrieval_latency': 0,
                    'llm_latency': 0
                }
            })
            mongo_logger.log_access(access_log_data)
            
            return result
        
        # 1. Recuperar documentos relevantes
        retrieved_docs = self.retriever.search(query, k=k)
        retrieval_latency = time.time() - retrieval_start
        
        # Log del servicio de retriever
        service_log_retriever = {
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc),  # CAMBIADO
            'service_type': 'pp1',
            'service_name': 'Retriever',
            'endpoint': 'local',
            'latency_ms': round(retrieval_latency * 1000, 2),
            'status_code': 200,
            'result': {
                'documents_retrieved': len(retrieved_docs),
                'query': query[:100]
            },
            'timeout': False,
            'error': None
        }
        mongo_logger.log_service(service_log_retriever)
        
        if not retrieved_docs:
            result = {
                'answer': 'No encontré información relevante en la normativa UFRO sobre este tema.',
                'citations': [],
                'latency': time.time() - start_time,
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': 0,
                'documents_retrieved': 0,
                'request_id': request_id,
                'success': True,
                'error': None
            }
            
            # Loggear métricas locales
            self._log_metrics(query, result)
            
            # Loggear en MongoDB (sin resultados)
            access_log_data.update({
                'decision': 'no_results',
                'timing_ms': round((time.time() - start_time) * 1000, 2),
                'status_code': 200,
                'errors': None,
                'pp1_used': True,
                'metadata': {
                    'retrieval_latency': round(retrieval_latency, 2),
                    'llm_latency': 0,
                    'documents_retrieved': 0
                }
            })
            mongo_logger.log_access(access_log_data)
            
            return result
        
        # 2. Generar contexto
        context = "\n\n".join([
            f"[Documento: {doc.get('title', 'Unknown')}, Página: {doc.get('page', 1)}]\n{doc.get('content', '')}"
            for doc in retrieved_docs
        ])
        
        # 3. Generar respuesta con el proveedor
        try:
            llm_start = time.time()
            
            provider = self.providers.get(provider_name)
            if not provider:
                # Fallback al primer proveedor disponible
                available_providers = list(self.providers.keys())
                if available_providers:
                    provider_name = available_providers[0]
                    provider = self.providers[provider_name]
                    logger.warning(f"Proveedor solicitado no disponible. Usando {provider_name} como fallback.")
                else:
                    result = {
                        'answer': '❌ No hay proveedores LLM disponibles en este momento.',
                        'citations': [],
                        'latency': time.time() - start_time,
                        'provider': 'none',
                        'retrieval_latency': round(retrieval_latency, 2),
                        'llm_latency': 0,
                        'documents_retrieved': len(retrieved_docs),
                        'request_id': request_id,
                        'success': False,
                        'error': 'no_providers_available'
                    }
                    
                    # Loggear métricas locales
                    self._log_metrics(query, result)
                    
                    # Loggear en MongoDB (error)
                    access_log_data.update({
                        'decision': 'error',
                        'timing_ms': round((time.time() - start_time) * 1000, 2),
                        'status_code': 500,
                        'errors': 'No hay proveedores LLM disponibles',
                        'pp1_used': True,
                        'metadata': {
                            'retrieval_latency': round(retrieval_latency, 2),
                            'llm_latency': 0,
                            'documents_retrieved': len(retrieved_docs)
                        }
                    })
                    mongo_logger.log_access(access_log_data)
                    
                    return result
            
            # Usar la función de prompt del sistema
            system_prompt = self.generate_system_prompt()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
            ]
            
            response = provider.chat(messages)
            llm_latency = time.time() - llm_start
            
            # Log del servicio LLM
            service_log_llm = {
                'request_id': request_id,
                'timestamp': datetime.now(timezone.utc),  # CAMBIADO
                'service_type': 'pp1',
                'service_name': f'LLM_{provider_name}',
                'endpoint': getattr(provider, 'endpoint', 'unknown'),
                'latency_ms': round(llm_latency * 1000, 2),
                'status_code': 200,
                'result': {
                    'response_length': len(response),
                    'provider': provider_name
                },
                'timeout': False,
                'error': None
            }
            mongo_logger.log_service(service_log_llm)
            
            # 4. Extraer citas
            citations = []
            for doc in retrieved_docs:
                citations.append(f"[{doc.get('title', 'Unknown')}, p.{doc.get('page', 1)}]")
            
            # 5. Crear resultado final
            total_latency = time.time() - start_time
            result = {
                'answer': response,
                'citations': citations[:3],
                'latency': round(total_latency, 2),
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': round(llm_latency, 2),
                'documents_retrieved': len(retrieved_docs),
                'request_id': request_id,
                'success': True,
                'error': None,
                'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
            }
            
            # 6. Loggear métricas locales
            self._log_metrics(query, result)
            
            # 7. Loggear en MongoDB (éxito)
            access_log_data.update({
                'decision': 'success',
                'timing_ms': round(total_latency * 1000, 2),
                'status_code': 200,
                'errors': None,
                'pp1_used': True,
                'metadata': {
                    'retrieval_latency': round(retrieval_latency, 2),
                    'llm_latency': round(llm_latency, 2),
                    'documents_retrieved': len(retrieved_docs),
                    'citations_count': len(citations[:3]),
                    'answer_length': len(response)
                }
            })
            mongo_logger.log_access(access_log_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando query: {e}")
            
            # Log del servicio LLM (error)
            service_log_llm = {
                'request_id': request_id,
                'timestamp': datetime.now(timezone.utc),  # CAMBIADO
                'service_type': 'pp1',
                'service_name': f'LLM_{provider_name}',
                'endpoint': getattr(provider, 'endpoint', 'unknown'),
                'latency_ms': round((time.time() - llm_start) * 1000, 2),
                'status_code': 500,
                'result': None,
                'timeout': False,
                'error': str(e)
            }
            mongo_logger.log_service(service_log_llm)
            
            result = {
                'answer': f'❌ Error procesando la consulta: {str(e)}',
                'citations': [],
                'latency': round(time.time() - start_time, 2),
                'provider': provider_name,
                'retrieval_latency': round(retrieval_latency, 2),
                'llm_latency': round(time.time() - llm_start, 2),
                'documents_retrieved': len(retrieved_docs),
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()  # CAMBIADO
            }
            
            # Loggear métricas locales incluso los errores
            self._log_metrics(query, result)
            
            # Loggear en MongoDB (error)
            access_log_data.update({
                'decision': 'error',
                'timing_ms': round((time.time() - start_time) * 1000, 2),
                'status_code': 500,
                'errors': str(e),
                'pp1_used': True,
                'metadata': {
                    'retrieval_latency': round(retrieval_latency, 2),
                    'llm_latency': round(time.time() - llm_start, 2),
                    'documents_retrieved': len(retrieved_docs),
                    'error_type': type(e).__name__
                }
            })
            mongo_logger.log_access(access_log_data)
            
            return result
    
    def _log_metrics(self, query: str, result: Dict):
        """Método helper para loggear métricas locales"""
        try:
            metrics_data = {
                'query': query,
                'provider': result['provider'],
                'answer': result['answer'],
                'latency': result['latency'],
                'retrieval_latency': result.get('retrieval_latency', 0),
                'llm_latency': result.get('llm_latency', 0),
                'citations': result['citations'],
                'documents_retrieved': result.get('documents_retrieved', 0),
                'request_id': result.get('request_id', ''),
                'success': result.get('success', False),
                'error': result.get('error')
            }
            metrics_logger.log_query(metrics_data)
        except Exception as e:
            logger.error(f"Error loggeando métricas locales: {e}")

# Instancia global del assistant
assistant = UFROAssistant()

# ==================== ENDPOINT PRINCIPAL PARA n8n ====================

@app.route('/api/query', methods=['POST'])
def api_query():
    """Endpoint principal para consultas desde n8n - Formato PP3"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos JSON',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 400
        
        query = data.get('query', '')
        provider = data.get('provider', 'chatgpt')
        k = data.get('k', 4)
        
        if not query:
            return jsonify({
                'error': 'Query vacía',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 400
        
        # Obtener información del usuario
        user_id = request.headers.get('X-User-Id', data.get('user_id', 'anonymous'))
        user_type = request.headers.get('X-User-Type', data.get('user_type', 'unknown'))
        
        # Procesar la consulta
        result = assistant.process_query(
            query=query, 
            provider_name=provider, 
            k=k,
            user_id=user_id, 
            user_type=user_type
        )
        
        # Formatear respuesta según PP3
        response_pp3 = format_response_pp3(result)
        
        # Agregar metadata adicional
        response_pp3['metadata']['user_type'] = user_type
        response_pp3['metadata']['mongo_connected'] = mongo_logger.is_connected()
        
        # Determinar status code
        status_code = 200 if result.get('success', False) else 500
        
        return jsonify(response_pp3), status_code
        
    except Exception as e:
        logger.error(f"Error en endpoint /api/query: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

# ==================== MODO CLI (para testing) ====================

def cli_mode():
    """Modo CLI mejorado para testing local"""
    parser = argparse.ArgumentParser(description='🤖 UFRO Assistant - Sistema RAG de Normativa')
    parser.add_argument('query', nargs='?', help='Consulta sobre normativa UFRO')
    parser.add_argument('--provider', choices=['chatgpt', 'deepseek'], default='deepseek', 
                       help='Proveedor LLM')
    parser.add_argument('--k', type=int, default=4, help='Número de documentos a recuperar')
    parser.add_argument('--user-id', help='ID del usuario para logging')
    parser.add_argument('--user-type', choices=['student', 'faculty', 'admin', 'external'], 
                       default='external', help='Tipo de usuario')
    parser.add_argument('--port', type=int, default=5001, help='Puerto para servidor API')
    parser.add_argument('--host', default='0.0.0.0', help='Host para servidor API')
    parser.add_argument('--serve', action='store_true', help='Iniciar servidor API')
    parser.add_argument('--health', action='store_true', help='Verificar estado del servicio')
    parser.add_argument('--metrics', action='store_true', help='Mostrar métricas recientes')
    
    args = parser.parse_args()
    
    if args.health:
        check_health()
        return
    
    if args.metrics:
        show_metrics_cli()
        return
    
    if args.serve:
        start_api_server(args.host, args.port)
        return
    
    if args.query:
        run_single_query(args)
    else:
        run_interactive_mode(args)

def check_health():
    """Verificar estado del servicio"""
    print("🔍 Verificando estado del servicio...")
    
    # Verificar MongoDB
    mongo_status = mongo_logger.is_connected()
    print(f"MongoDB: {'✅ Conectado' if mongo_status else '❌ Desconectado'}")
    
    # Verificar assistant
    assistant_status = assistant.is_initialized()
    print(f"Assistant: {'✅ Inicializado' if assistant_status else '❌ No inicializado'}")
    
    if assistant_status:
        print(f"Proveedores disponibles: {list(assistant.providers.keys())}")
    
    # Verificar endpoints
    print(f"\n📡 Endpoints disponibles:")
    print(f"  POST /api/query         - Procesar consulta (Formato PP3)")
    print(f"  GET  /health            - Health check")
    print(f"  GET  /api/mongo/status  - Estado MongoDB")
    print(f"  GET  /api/mongo/metrics/summary - Métricas resumen")
    print(f"  GET  /api/mongo/metrics/recent  - Logs recientes")
    print(f"  GET  /api/metrics/recent - Métricas locales")
    print(f"  GET  /api/metrics/stats  - Estadísticas proveedores")

def show_metrics_cli():
    """Mostrar métricas en CLI"""
    print("📊 MÉTRICAS DEL SISTEMA")
    print("=" * 60)
    
    # Métricas de MongoDB
    if mongo_logger.is_connected():
        print("\n📈 MongoDB Metrics:")
        metrics = mongo_logger.get_metrics_summary(1)  # Último día
        for metric in metrics:
            print(f"  {metric['_id']}: {metric['total_requests']} requests, "
                  f"avg {metric['avg_latency_ms']:.1f}ms, "
                  f"success {metric['success_rate']*100:.1f}%")
    else:
        print("❌ MongoDB no conectado para métricas")
    
    # Métricas locales
    print("\n📊 Métricas Locales:")
    try:
        stats = metrics_logger.get_provider_stats()
        for stat in stats:
            print(f"  {stat['provider']}: {stat['total_queries']} queries, "
                  f"avg {stat['avg_latency']:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")

def start_api_server(host, port):
    """Iniciar servidor API"""
    print(f"🚀 Iniciando servidor API UFRO Assistant")
    print(f"📡 Host: {host}")
    print(f"🔌 Puerto: {port}")
    print(f"📊 MongoDB: {'✅ Conectado' if mongo_logger.is_connected() else '❌ Desconectado'}")
    print(f"🤖 Assistant: {'✅ Inicializado' if assistant.is_initialized() else '❌ No inicializado'}")
    print("\n🔗 Endpoints disponibles:")
    print(f"  POST http://localhost:{port}/api/query  (Formato PP3)")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  GET  http://localhost:{port}/api/mongo/status")
    print(f"\n📝 Ejemplo para n8n (Formato PP3):")
    print(f'  curl -X POST http://localhost:{port}/api/query \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -H "X-User-Id: estudiante001" \\')
    print('    -H "X-User-Type: student" \\')
    print('    -d \'{"query": "¿Cuántos días tengo para retractarme?", "provider": "chatgpt"}\'')
    
    app.run(host=host, port=port, debug=False)

def run_interactive_mode(args):
    """Modo interactivo para testing"""
    print("🤖 UFRO Assistant - Modo interactivo")
    print("Escriba 'salir' para terminar\n")
    
    while True:
        try:
            query = input("Consulta: ").strip()
            if query.lower() in ['salir', 'exit', 'quit']:
                break
            
            if query:
                result = assistant.process_query(
                    query, 
                    args.provider, 
                    args.k,
                    user_id=args.user_id or 'cli_user',
                    user_type=args.user_type or 'external'
                )
                
                print(f"\n🤖 Respuesta ({result['provider']}):")
                print(result['answer'])
                
                # Mostrar citas en formato PP3
                if result['citations']:
                    print(f"\n📚 Citaciones (PP3):")
                    formatted_citations = format_citations_pp3(result['citations'])
                    for cite in formatted_citations[:2]:  # Solo mostrar 2 en modo interactivo
                        print(f"  • Doc: {cite['doc']}, Pág: {cite['page']}")
                
                print(f"\n⏱️  Métricas:")
                print(f"  • Tiempo: {result['latency']:.2f}s ({result['latency']*1000:.0f}ms)")
                print(f"  • Decision: {'success' if result.get('success') else 'error'}")
                print(f"  • Documents: {result['documents_retrieved']}")
                print(f"  • ID: {result.get('request_id', 'N/A')[:8]}...")
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_single_query(args):
    """Ejecutar una consulta única desde CLI con formato PP3"""
    print(f"\n🔍 UFRO Assistant - Procesando consulta...")
    print(f"📝 Query: {args.query}")
    print(f"🤖 Provider: {args.provider}")
    print(f"📚 k: {args.k}")
    print(f"👤 User: {args.user_id or 'cli_user'} ({args.user_type})")
    print("-" * 60)
    
    result = assistant.process_query(
        query=args.query,
        provider_name=args.provider,
        k=args.k,
        user_id=args.user_id or 'cli_user',
        user_type=args.user_type or 'external'
    )
    
    # Mostrar respuesta
    print(f"\n🤖 RESPUESTA NORMATIVA UFRO ({result['provider'].upper()}):")
    print("─" * 60)
    print(result['answer'])
    print("─" * 60)
    
    # Mostrar citas en formato PP3
    if result['citations']:
        print(f"\n📚 CITAS (Formato PP3):")
        formatted_citations = format_citations_pp3(result['citations'])
        for i, cite in enumerate(formatted_citations[:3]):
            print(f"  {i+1}. Documento: {cite['doc']}, Página: {cite['page']}")
    
    # Métricas en formato analítica
    print(f"\n📊 MÉTRICAS PARA ANALÍTICA:")
    print(f"  • Decision: {'success' if result.get('success') else 'error'}")
    print(f"  • Timing: {result['latency']:.2f}s ({result['retrieval_latency']:.2f}s retrieval, {result['llm_latency']:.2f}s LLM)")
    print(f"  • Timing (ms): {result['latency']*1000:.1f}ms")
    print(f"  • Documents retrieved: {result['documents_retrieved']}")
    print(f"  • Request ID: {result.get('request_id', 'N/A')}")
    print(f"  • Status: {'✅ Success' if result.get('success') else '❌ Error'}")
    print(f"  • User Type: {args.user_type}")
    
    # Mostrar también formato JSON PP3 para n8n
    print(f"\n🔗 FORMATO JSON PP3 (para n8n/integraciones):")
    pp3_response = format_response_pp3(result)
    pp3_response['metadata']['user_type'] = args.user_type
    
    print(json.dumps(pp3_response, indent=2, ensure_ascii=False))
    
    # Información sobre logging
    print(f"\n📝 LOGGING:")
    print(f"  • MongoDB: {'✅ Logs guardados' if mongo_logger.is_connected() else '❌ No conectado'}")
    if mongo_logger.is_connected():
        print(f"  • Log ID: {result.get('request_id', 'N/A')}")

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    print("🚀 Iniciando servidor Flask en 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)