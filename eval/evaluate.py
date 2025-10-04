import json
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.cost_rates = {
            'chatgpt': 0.005,  # USD por consulta promedio
            'deepseek': 0.002  # USD por consulta promedio
        }
    
    def comprehensive_evaluation(self, prediction: Dict, gold: Dict) -> Dict:
        """Evaluación completa con todas las métricas"""
        return {
            # Calidad de respuesta
            'exact_match': self.exact_match(prediction['answer'], gold.get('expected_answer', '')),
            'semantic_similarity': self.semantic_similarity(prediction['answer'], gold.get('expected_answer', '')),
            'answer_length': len(prediction['answer'].split()),
            
            # Citas y referencias
            'citation_coverage': self.citation_coverage(prediction['answer']),
            'citation_accuracy': self.citation_accuracy(prediction['citations'], gold.get('expected_citations', [])),
            'citation_count': len(prediction['citations']),
            
            # Rendimiento
            'total_latency': prediction.get('latency', 0),
            'retrieval_latency': prediction.get('retrieval_latency', 0),
            'llm_latency': prediction.get('llm_latency', 0),
            
            # Costo estimado
            'estimated_cost': self.estimate_cost(prediction.get('provider', ''), prediction['answer'])
        }
    
    def estimate_cost(self, provider: str, answer: str) -> float:
        """Estima costo basado en tokens de respuesta"""
        if provider not in self.cost_rates:
            return 0.0
        
        # Estimación simple: ~1 token por palabra
        word_count = len(answer.split())
        return self.cost_rates[provider] * (word_count / 500)  # Normalizado a 500 palabras
    
    def citation_accuracy(self, actual_citations: List[str], expected_citations: List[str]) -> float:
        """Precisión de citas vs las esperadas"""
        if not expected_citations:
            return 1.0  # Si no hay citas esperadas, se considera correcto
        
        if not actual_citations:
            return 0.0
        
        # Verificar si las citas esperadas están en las actuales
        matches = 0
        for expected in expected_citations:
            if any(expected.lower() in actual.lower() for actual in actual_citations):
                matches += 1
        
        return matches / len(expected_citations)

def generate_comparative_table(results_chatgpt: List[Dict], results_deepseek: List[Dict]) -> str:
    """Genera tabla comparativa profesional"""
    from tabulate import tabulate
    
    # Calcular métricas agregadas
    def aggregate_metrics(results):
        return {
            'Exact Match': np.mean([r['exact_match'] for r in results]),
            'Similitud Semántica': np.mean([r['semantic_similarity'] for r in results]),
            'Cobertura Citas': np.mean([r['citation_coverage'] for r in results]),
            'Latencia (s)': np.mean([r['total_latency'] for r in results]),
            'Costo Promedio (USD)': np.mean([r.get('estimated_cost', 0) for r in results]),
            'Muestras': len(results)
        }
    
    gpt_metrics = aggregate_metrics(results_chatgpt)
    ds_metrics = aggregate_metrics(results_deepseek)
    
    table_data = []
    for metric in ['Exact Match', 'Similitud Semántica', 'Cobertura Citas', 'Latencia (s)', 'Costo Promedio (USD)']:
        table_data.append([
            metric,
            f"{gpt_metrics[metric]:.3f}" if metric != 'Latencia (s)' else f"{gpt_metrics[metric]:.2f}",
            f"{ds_metrics[metric]:.3f}" if metric != 'Latencia (s)' else f"{ds_metrics[metric]:.2f}",
            f"{(ds_metrics[metric] - gpt_metrics[metric]):.3f}" if metric != 'Muestras' else ""
        ])
    
    table = tabulate(table_data, 
                    headers=['Métrica', 'ChatGPT', 'DeepSeek', 'Diferencia'],
                    tablefmt='grid',
                    floatfmt=".3f")
    
    # Hallazgos clave
    findings = []
    if ds_metrics['Costo Promedio (USD)'] < gpt_metrics['Costo Promedio (USD)']:
        savings = (gpt_metrics['Costo Promedio (USD)'] - ds_metrics['Costo Promedio (USD)']) / gpt_metrics['Costo Promedio (USD)'] * 100
        findings.append(f"• DeepSeek es {savings:.1f}% más económico que ChatGPT")
    
    if ds_metrics['Latencia (s)'] < gpt_metrics['Latencia (s)']:
        findings.append(f"• DeepSeek es {gpt_metrics['Latencia (s)']/ds_metrics['Latencia (s)']:.1f}x más rápido")
    
    if gpt_metrics['Exact Match'] > ds_metrics['Exact Match']:
        findings.append(f"• ChatGPT tiene mayor exactitud (+{(gpt_metrics['Exact Match']-ds_metrics['Exact Match'])*100:.1f}%)")
    
    report = f"""
📊 TABLA COMPARATIVA - UFRO ASSISTANT
{table}

🔍 HALLAZGOS PRINCIPALES:
{chr(10).join(findings)}

💡 RECOMENDACIONES:
1. Usar DeepSeek para consultas rutinarias (mejor costo/performance)
2. Usar ChatGPT para consultas críticas (mayor exactitud)
3. Mantener ambos proveedores para redundancia
"""
    
    return report

def evaluate_response(prediction: str, gold: str = None) -> Dict:
    """Función helper para evaluación rápida"""
    evaluator = Evaluator()
    
    if gold:
        return {
            'has_citations': evaluator.citation_coverage(prediction),
            'similarity': evaluator.semantic_similarity(prediction, gold) if gold else 0
        }
    
    return {'has_citations': evaluator.citation_coverage(prediction)}
