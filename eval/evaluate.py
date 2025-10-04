# eval/evaluate.py
import json
import time
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def evaluate_response(self, prediction: Dict, gold: Dict) -> Dict:
        """
        Evalúa una respuesta comparándola con la respuesta esperada
        
        Args:
            prediction: Dict con la respuesta del sistema
            gold: Dict con la respuesta esperada y metadata
            
        Returns:
            Dict con métricas de evaluación
        """
        try:
            # Métricas básicas
            exact_match = self.exact_match(prediction['answer'], gold.get('expected_answer', ''))
            semantic_similarity = self.semantic_similarity(prediction['answer'], gold.get('expected_answer', ''))
            citation_coverage = self.citation_coverage(prediction['answer'])
            
            return {
                'exact_match': exact_match,
                'semantic_similarity': semantic_similarity,
                'citation_coverage': citation_coverage,
                'has_answer': len(prediction['answer'].strip()) > 0,
                'answer_length': len(prediction['answer'])
            }
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            return {
                'exact_match': 0.0,
                'semantic_similarity': 0.0,
                'citation_coverage': 0.0,
                'has_answer': False,
                'answer_length': 0
            }
    
    def exact_match(self, pred: str, gold: str) -> float:
        """Calcula exact match simplificado"""
        if not pred or not gold:
            return 0.0
        
        pred_lower = pred.lower().strip()
        gold_lower = gold.lower().strip()
        
        # Match parcial basado en palabras clave importantes
        gold_words = set(gold_lower.split())
        pred_words = set(pred_lower.split())
        
        if not gold_words:
            return 0.0
            
        # Calcular intersección de palabras significativas
        common_words = gold_words.intersection(pred_words)
        return len(common_words) / len(gold_words)
    
    def semantic_similarity(self, pred: str, gold: str) -> float:
        """Calcula similitud semántica usando embeddings"""
        if not pred or not gold:
            return 0.0
        
        try:
            emb1 = self.model.encode([pred])
            emb2 = self.model.encode([gold])
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"❌ Error calculando similitud semántica: {e}")
            return 0.0
    
    def citation_coverage(self, response: str) -> float:
        """Verifica si hay citas en la respuesta"""
        import re
        # Buscar patrones de citas [Documento, p.X] o [Documento, página X]
        citations = re.findall(r'\[.*?[pP](?:ágina|\.)\s*\d+.*?\]', response)
        return 1.0 if len(citations) > 0 else 0.0
    
    def evaluate_batch(self, predictions: List[Dict], gold_standard: List[Dict]) -> Dict:
        """Evalúa un lote de respuestas"""
        if len(predictions) != len(gold_standard):
            raise ValueError("El número de predicciones y estándares dorados debe ser igual")
        
        results = []
        for pred, gold in zip(predictions, gold_standard):
            metrics = self.evaluate_response(pred, gold)
            results.append(metrics)
        
        # Métricas agregadas
        aggregated = {
            'num_samples': len(results),
            'avg_exact_match': np.mean([r['exact_match'] for r in results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in results]),
            'avg_citation_coverage': np.mean([r['citation_coverage'] for r in results]),
            'answer_rate': np.mean([r['has_answer'] for r in results]),
            'avg_answer_length': np.mean([r['answer_length'] for r in results])
        }
        
        return {
            'individual_results': results,
            'aggregated_metrics': aggregated
        }

# Función helper para compatibilidad
def evaluate_response(prediction: str, gold: str = None) -> Dict:
    """Función helper para evaluación rápida"""
    evaluator = Evaluator()
    
    if gold:
        return {
            'has_citations': evaluator.citation_coverage(prediction),
            'similarity': evaluator.semantic_similarity(prediction, gold) if gold else 0
        }
    
    return {'has_citations': evaluator.citation_coverage(prediction)}