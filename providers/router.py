from typing import List, Dict

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

class ProviderRouter:
    def __init__(self):
        self.providers = {
            'chatgpt': ChatGPTProvider(),
            'deepseek': DeepSeekProvider()
        }
        self.stats = {
            'chatgpt': {'calls': 0, 'errors': 0, 'total_latency': 0},
            'deepseek': {'calls': 0, 'errors': 0, 'total_latency': 0}
        }
    
    def chat(self, messages: List[Dict], provider_name: str = None, **kwargs):
        import time
        
        # Si no se especifica proveedor, usar el más rápido históricamente
        if provider_name is None:
            provider_name = self._get_fastest_provider()
        
        start_time = time.time()
        
        try:
            provider = self.providers[provider_name]
            response = provider.chat(messages, **kwargs)
            
            # Actualizar estadísticas
            latency = time.time() - start_time
            self.stats[provider_name]['calls'] += 1
            self.stats[provider_name]['total_latency'] += latency
            
            # Log detallado
            print(f"✅ {provider_name.upper()} - Latencia: {latency:.2f}s - Tokens: ~{len(response.split())}")
            
            return response, provider_name, latency
            
        except Exception as e:
            self.stats[provider_name]['errors'] += 1
            print(f"❌ Error en {provider_name}: {e}")
            
            # Fallback al otro proveedor
            fallback = 'deepseek' if provider_name == 'chatgpt' else 'chatgpt'
            print(f"🔄 Fallback a {fallback}")
            return self.chat(messages, fallback, **kwargs)
    
    def _get_fastest_provider(self):
        """Retorna el proveedor con menor latencia promedio"""
        if self.stats['chatgpt']['calls'] == 0:
            return 'deepseek'  # Default más económico
        
        gpt_avg = self.stats['chatgpt']['total_latency'] / self.stats['chatgpt']['calls']
        ds_avg = self.stats['deepseek']['total_latency'] / self.stats['deepseek']['calls']
        
        return 'deepseek' if ds_avg < gpt_avg else 'chatgpt'
    
    def get_stats(self):
        """Retorna estadísticas para reportes"""
        stats = {}
        for provider, data in self.stats.items():
            if data['calls'] > 0:
                stats[provider] = {
                    'calls': data['calls'],
                    'errors': data['errors'],
                    'avg_latency': data['total_latency'] / data['calls'],
                    'error_rate': data['errors'] / data['calls']
                }
        return stats