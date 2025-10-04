"""
Módulo de métricas de rendimiento para H7 - Costo y latencia
Mide latencia end-to-end y por etapa, estima costos por consulta
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import tiktoken

from providers.base import Provider
from rag.retrieve import RAGRetriever


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento para una consulta"""
    query: str
    provider_name: str
    
    # Métricas de tiempo (en segundos)
    retrieval_time: float
    llm_time: float
    total_time: float
    
    # Métricas de tokens y costo
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    
    # Métricas adicionales
    num_retrieved_docs: int
    retrieval_success: bool
    generation_success: bool
    error_message: Optional[str] = None


class PerformanceEvaluator:
    """Evaluador de rendimiento para sistemas RAG"""
    
    # Precios públicos por 1K tokens (actualizado 2024)
    PRICING = {
        'chatgpt': {
            'input': 0.0015,   # GPT-3.5-turbo input
            'output': 0.002,   # GPT-3.5-turbo output
        },
        'deepseek': {
            'input': 0.00014,  # DeepSeek input estimado
            'output': 0.00028, # DeepSeek output estimado
        }
    }
    
    def __init__(self):
        """Inicializa el evaluador de rendimiento"""
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Para contar tokens
        
    def evaluate_query_performance(
        self,
        query: str,
        provider: Provider,
        retriever: RAGRetriever
    ) -> PerformanceMetrics:
        """
        Evalúa el rendimiento de una consulta individual
        
        Args:
            query: Consulta a evaluar
            provider: Provider LLM a usar
            retriever: Sistema de recuperación RAG
            
        Returns:
            Métricas de rendimiento completas
        """
        start_total = time.time()
        
        # Etapa 1: Recuperación
        retrieval_success = False
        retrieved_docs = []
        error_message = None
        
        start_retrieval = time.time()
        try:
            retrieved_docs = retriever.retrieve(query)
            retrieval_success = True
        except Exception as e:
            error_message = f"Error en recuperación: {str(e)}"
        retrieval_time = time.time() - start_retrieval
        
        # Etapa 2: Generación LLM
        generation_success = False
        response = ""
        input_tokens = 0
        output_tokens = 0
        
        start_llm = time.time()
        try:
            if retrieval_success:
                response = provider.generate_response(query, retrieved_docs)
                generation_success = True
                
                # Contar tokens
                input_tokens = self._count_input_tokens(query, retrieved_docs)
                output_tokens = self._count_output_tokens(response)
            else:
                error_message = error_message or "Fallo en recuperación"
        except Exception as e:
            if not error_message:
                error_message = f"Error en generación: {str(e)}"
        llm_time = time.time() - start_llm
        
        total_time = time.time() - start_total
        total_tokens = input_tokens + output_tokens
        
        # Calcular costo estimado
        estimated_cost = self._estimate_cost(
            provider.name, input_tokens, output_tokens
        )
        
        return PerformanceMetrics(
            query=query,
            provider_name=provider.name,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            total_time=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            num_retrieved_docs=len(retrieved_docs),
            retrieval_success=retrieval_success,
            generation_success=generation_success,
            error_message=error_message
        )
    
    def evaluate_batch_performance(
        self,
        queries: List[str],
        provider: Provider,
        retriever: RAGRetriever
    ) -> List[PerformanceMetrics]:
        """
        Evalúa el rendimiento de un conjunto de consultas
        
        Args:
            queries: Lista de consultas a evaluar
            provider: Provider LLM a usar
            retriever: Sistema de recuperación RAG
            
        Returns:
            Lista de métricas de rendimiento
        """
        results = []
        
        print(f"Evaluando rendimiento de {len(queries)} consultas con {provider.name}...")
        
        for i, query in enumerate(queries):
            print(f"Procesando consulta {i+1}/{len(queries)}: {query[:50]}...")
            
            metrics = self.evaluate_query_performance(query, provider, retriever)
            results.append(metrics)
            
            # Mostrar progreso
            if metrics.generation_success:
                print(f"  ✅ Tiempo total: {metrics.total_time:.2f}s, Costo: ${metrics.estimated_cost_usd:.6f}")
            else:
                print(f"  ❌ Error: {metrics.error_message}")
        
        return results
    
    def save_performance_csv(
        self,
        metrics_list: List[PerformanceMetrics],
        output_path: str = "eval/performance_metrics.csv"
    ):
        """
        Guarda las métricas de rendimiento en CSV
        
        Args:
            metrics_list: Lista de métricas a guardar
            output_path: Ruta del archivo CSV
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'query', 'provider_name', 'retrieval_time', 'llm_time', 'total_time',
            'input_tokens', 'output_tokens', 'total_tokens', 'estimated_cost_usd',
            'num_retrieved_docs', 'retrieval_success', 'generation_success', 'error_message'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in metrics_list:
                writer.writerow(asdict(metrics))
        
        print(f"Métricas de rendimiento guardadas en {output_path}")
    
    def generate_performance_report(
        self,
        metrics_list: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """
        Genera un reporte agregado de rendimiento
        
        Args:
            metrics_list: Lista de métricas
            
        Returns:
            Diccionario con estadísticas agregadas
        """
        if not metrics_list:
            return {}
        
        # Filtrar solo consultas exitosas
        successful = [m for m in metrics_list if m.generation_success]
        failed = [m for m in metrics_list if not m.generation_success]
        
        if not successful:
            return {
                'total_queries': len(metrics_list),
                'successful_queries': 0,
                'failed_queries': len(failed),
                'success_rate': 0.0
            }
        
        # Estadísticas de tiempo
        retrieval_times = [m.retrieval_time for m in successful]
        llm_times = [m.llm_time for m in successful]
        total_times = [m.total_time for m in successful]
        
        # Estadísticas de tokens y costo
        input_tokens = [m.input_tokens for m in successful]
        output_tokens = [m.output_tokens for m in successful]
        total_tokens = [m.total_tokens for m in successful]
        costs = [m.estimated_cost_usd for m in successful]
        
        report = {
            'total_queries': len(metrics_list),
            'successful_queries': len(successful),
            'failed_queries': len(failed),
            'success_rate': len(successful) / len(metrics_list),
            
            'timing_stats': {
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times),
                'avg_llm_time': sum(llm_times) / len(llm_times),
                'avg_total_time': sum(total_times) / len(total_times),
                'max_total_time': max(total_times),
                'min_total_time': min(total_times)
            },
            
            'token_stats': {
                'avg_input_tokens': sum(input_tokens) / len(input_tokens),
                'avg_output_tokens': sum(output_tokens) / len(output_tokens),
                'avg_total_tokens': sum(total_tokens) / len(total_tokens),
                'total_input_tokens': sum(input_tokens),
                'total_output_tokens': sum(output_tokens)
            },
            
            'cost_stats': {
                'avg_cost_per_query': sum(costs) / len(costs),
                'total_estimated_cost': sum(costs),
                'max_cost_per_query': max(costs),
                'min_cost_per_query': min(costs)
            }
        }
        
        return report
    
    def _count_input_tokens(self, query: str, retrieved_docs: List) -> int:
        """Cuenta tokens de entrada (query + documentos recuperados)"""
        try:
            # Contar tokens de la consulta
            query_tokens = len(self.encoding.encode(query))
            
            # Contar tokens de documentos recuperados
            docs_text = "\n".join([doc.content for doc in retrieved_docs])
            docs_tokens = len(self.encoding.encode(docs_text))
            
            # Agregar overhead del prompt del sistema (estimado)
            system_prompt_tokens = 200  # Estimación conservadora
            
            return query_tokens + docs_tokens + system_prompt_tokens
        except Exception:
            return 0
    
    def _count_output_tokens(self, response: str) -> int:
        """Cuenta tokens de salida (respuesta generada)"""
        try:
            return len(self.encoding.encode(response))
        except Exception:
            return 0
    
    def _estimate_cost(self, provider_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo de una consulta basado en precios públicos"""
        provider_key = provider_name.lower()
        
        if provider_key not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[provider_key]
        
        # Cálculo: (tokens / 1000) * precio_por_1k_tokens
        input_cost = (input_tokens / 1000.0) * pricing['input']
        output_cost = (output_tokens / 1000.0) * pricing['output']
        
        return input_cost + output_cost
    
    def print_performance_report(self, report: Dict[str, Any]):
        """Imprime un reporte de rendimiento formateado"""
        print("\n" + "="*60)
        print("REPORTE DE RENDIMIENTO (H7)")
        print("="*60)
        
        print(f"Total de consultas: {report['total_queries']}")
        print(f"Consultas exitosas: {report['successful_queries']}")
        print(f"Consultas fallidas: {report['failed_queries']}")
        print(f"Tasa de éxito: {report['success_rate']:.1%}")
        
        if 'timing_stats' in report:
            timing = report['timing_stats']
            print(f"\n📊 ESTADÍSTICAS DE TIEMPO:")
            print(f"  Tiempo promedio recuperación: {timing['avg_retrieval_time']:.3f}s")
            print(f"  Tiempo promedio LLM: {timing['avg_llm_time']:.3f}s")
            print(f"  Tiempo promedio total: {timing['avg_total_time']:.3f}s")
            print(f"  Tiempo máximo: {timing['max_total_time']:.3f}s")
            print(f"  Tiempo mínimo: {timing['min_total_time']:.3f}s")
        
        if 'token_stats' in report:
            tokens = report['token_stats']
            print(f"\n🔤 ESTADÍSTICAS DE TOKENS:")
            print(f"  Promedio tokens entrada: {tokens['avg_input_tokens']:.0f}")
            print(f"  Promedio tokens salida: {tokens['avg_output_tokens']:.0f}")
            print(f"  Promedio tokens total: {tokens['avg_total_tokens']:.0f}")
            print(f"  Total tokens procesados: {tokens['total_input_tokens'] + tokens['total_output_tokens']:,}")
        
        if 'cost_stats' in report:
            costs = report['cost_stats']
            print(f"\n💰 ESTADÍSTICAS DE COSTO:")
            print(f"  Costo promedio por consulta: ${costs['avg_cost_per_query']:.6f}")
            print(f"  Costo total estimado: ${costs['total_estimated_cost']:.6f}")
            print(f"  Costo máximo por consulta: ${costs['max_cost_per_query']:.6f}")
            print(f"  Costo mínimo por consulta: ${costs['min_cost_per_query']:.6f}")


def main():
    """Función principal para testing del módulo"""
    import argparse
    from dotenv import load_dotenv
    import os
    from providers.chatgpt import ChatGPTProvider
    from providers.deepseek import DeepSeekProvider
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluar rendimiento del sistema RAG (H7)")
    parser.add_argument("--provider", choices=["chatgpt", "deepseek"], required=True)
    parser.add_argument("--queries", nargs='+', default=[
        "¿Qué es machine learning?",
        "Explica la diferencia entre supervised y unsupervised learning",
        "¿Cómo funciona una red neuronal?"
    ])
    parser.add_argument("--output", default="eval/performance_metrics.csv")
    
    args = parser.parse_args()
    
    # Crear provider
    if args.provider == "chatgpt":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY requerida")
        provider = ChatGPTProvider(api_key=api_key)
    else:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY requerida")
        provider = DeepSeekProvider(api_key=api_key)
    
    # Crear retriever
    retriever = RAGRetriever(top_k=5)
    
    # Evaluar rendimiento
    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate_batch_performance(args.queries, provider, retriever)
    
    # Guardar y mostrar resultados
    evaluator.save_performance_csv(metrics, args.output)
    
    report = evaluator.generate_performance_report(metrics)
    evaluator.print_performance_report(report)


if __name__ == "__main__":
    main()