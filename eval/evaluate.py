import json
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def exact_match(self, pred: str, gold: str) -> float:
        """Calcula exact match simplificado"""
        pred_lower = pred.lower().strip()
        gold_lower = gold.lower().strip()
        
        # Match parcial si contiene palabras clave
        key_words = gold_lower.split()[:5]  # Primeras 5 palabras
        matches = sum(1 for word in key_words if word in pred_lower)
        
        return matches / len(key_words) if key_words else 0
    
    def semantic_similarity(self, pred: str, gold: str) -> float:
        """Calcula similitud semántica"""
        emb1 = self.model.encode([pred])
        emb2 = self.model.encode([gold])
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def citation_coverage(self, response: str) -> float:
        """Verifica si hay citas en la respuesta"""
        # Buscar patrones de citas [Doc, p.X]
        import re
        citations = re.findall(r'\[.*?p\.\d+\]', response)
        
        return 1.0 if len(citations) > 0 else 0.0
    
    def evaluate_response(self, prediction: Dict, gold: Dict) -> Dict:
        """Evalúa una respuesta individual"""
        metrics = {
            'exact_match': self.exact_match(prediction['answer'], gold.get('expected_answer', '')),
            'semantic_similarity': self.semantic_similarity(prediction['answer'], gold.get('expected_answer', '')),
            'citation_coverage': self.citation_coverage(prediction['answer']),
            'latency': prediction.get('latency', 0)
        }
        
        return metrics

def evaluate_response(prediction: str, gold: str = None) -> Dict:
    """Función helper para evaluación rápida"""
    evaluator = Evaluator()
    
    if gold:
        return {
            'has_citations': evaluator.citation_coverage(prediction),
            'similarity': evaluator.semantic_similarity(prediction, gold) if gold else 0
        }
    
    return {'has_citations': evaluator.citation_coverage(prediction)}
