# metrics_logger.py
import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os

class MetricsLogger:
    def __init__(self, log_dir: str = "logs/reportes"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de logs
        self.metrics_file = self.log_dir / "query_metrics.jsonl"
        self.stats_file = self.log_dir / "provider_stats.csv"
        self.comparative_file = self.log_dir / "tabla_comparativa.html"
        
        # Tasas de costo (USD por 1000 tokens)
        self.cost_rates = {
            'chatgpt': {'input': 0.0015, 'output': 0.002},
            'deepseek': {'input': 0.00014, 'output': 0.00028}
        }
    
    def log_query(self, query_data: Dict):
        """Registra una consulta con sus métricas"""
        timestamp = datetime.now().isoformat()
        
        # Estimar tokens y costo
        estimated_tokens = self._estimate_tokens(query_data.get('answer', ''))
        cost = self._calculate_cost(
            query_data.get('provider', ''),
            estimated_tokens,
            query_data.get('retrieval_latency', 0)
        )
        
        log_entry = {
            'timestamp': timestamp,
            'query': query_data.get('query', ''),
            'provider': query_data.get('provider', ''),
            'answer_length': len(query_data.get('answer', '')),
            'estimated_tokens': estimated_tokens,
            'latency_total': query_data.get('latency', 0),
            'latency_retrieval': query_data.get('retrieval_latency', 0),
            'latency_llm': query_data.get('llm_latency', 0),
            'estimated_cost_usd': cost,
            'citations_count': len(query_data.get('citations', [])),
            'documents_retrieved': query_data.get('documents_retrieved', 0)
        }
        
        # Guardar en JSONL
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Actualizar estadísticas
        self._update_provider_stats(log_entry)
        
        return log_entry
    
    def _estimate_tokens(self, text: str) -> int:
        """Estima tokens basado en longitud del texto"""
        # Aproximación: 1 token ≈ 4 caracteres en español
        return len(text) // 4
    
    def _calculate_cost(self, provider: str, tokens: int, retrieval_latency: float) -> float:
        """Calcula costo estimado de la consulta"""
        if provider not in self.cost_rates:
            return 0.0
        
        # Costo de generación (asumiendo 500 tokens de input + output)
        input_cost = (500 * self.cost_rates[provider]['input']) / 1000
        output_cost = (tokens * self.cost_rates[provider]['output']) / 1000
        
        # Costo de infraestructura (estimado)
        infra_cost = (retrieval_latency * 0.0001)  # Costo aproximado por segundo
        
        return round(input_cost + output_cost + infra_cost, 6)
    
    def _update_provider_stats(self, log_entry: Dict):
        """Actualiza estadísticas por proveedor"""
        try:
            # Leer estadísticas existentes
            stats = {}
            if self.stats_file.exists():
                df = pd.read_csv(self.stats_file)
                stats = df.set_index('provider').to_dict('index')
            
            provider = log_entry['provider']
            if provider not in stats:
                stats[provider] = {
                    'total_queries': 0,
                    'avg_latency': 0,
                    'avg_cost': 0,
                    'total_cost': 0,
                    'avg_tokens': 0,
                    'success_rate': 0
                }
            
            # Actualizar estadísticas
            current = stats[provider]
            n = current['total_queries']
            
            stats[provider]['total_queries'] = n + 1
            stats[provider]['avg_latency'] = (current['avg_latency'] * n + log_entry['latency_total']) / (n + 1)
            stats[provider]['avg_cost'] = (current['avg_cost'] * n + log_entry['estimated_cost_usd']) / (n + 1)
            stats[provider]['total_cost'] = current['total_cost'] + log_entry['estimated_cost_usd']
            stats[provider]['avg_tokens'] = (current['avg_tokens'] * n + log_entry['estimated_tokens']) / (n + 1)
            
            # Guardar
            df_stats = pd.DataFrame.from_dict(stats, orient='index')
            df_stats.reset_index(inplace=True)
            df_stats.rename(columns={'index': 'provider'}, inplace=True)
            df_stats.to_csv(self.stats_file, index=False)
            
        except Exception as e:
            print(f"⚠️ Error actualizando estadísticas: {e}")
    
    def generate_comparative_table(self) -> str:
        """Genera tabla comparativa HTML con conclusiones"""
        try:
            if not self.metrics_file.exists():
                return "<p>No hay datos suficientes para generar tabla comparativa</p>"
            
            # Cargar datos
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = [json.loads(line) for line in f if line.strip()]
            
            if not metrics_data:
                return "<p>No hay consultas registradas</p>"
            
            df = pd.DataFrame(metrics_data)
            
            # Calcular métricas por proveedor
            comparative_data = []
            for provider in df['provider'].unique():
                provider_data = df[df['provider'] == provider]
                
                comparative_data.append({
                    'Proveedor': provider.upper(),
                    'Consultas': len(provider_data),
                    'Latencia Promedio (s)': f"{provider_data['latency_total'].mean():.2f}",
                    'Costo Promedio (USD)': f"${provider_data['estimated_cost_usd'].mean():.6f}",
                    'Costo Total (USD)': f"${provider_data['estimated_cost_usd'].sum():.6f}",
                    'Tokens Promedio': f"{provider_data['estimated_tokens'].mean():.0f}",
                    'Citas Promedio': f"{provider_data['citations_count'].mean():.1f}",
                    'Eficiencia Costo/Latencia': f"${(provider_data['estimated_cost_usd'].mean() / provider_data['latency_total'].mean()):.6f}/s"
                })
            
            # Generar conclusiones
            conclusions = self._generate_conclusions(df)
            
            # Crear tabla HTML
            html_table = self._create_html_table(comparative_data, conclusions)
            
            # Guardar archivo
            with open(self.comparative_file, 'w', encoding='utf-8') as f:
                f.write(html_table)
            
            return html_table
            
        except Exception as e:
            return f"<p>Error generando tabla: {str(e)}</p>"
    
    def _generate_conclusions(self, df: pd.DataFrame) -> List[str]:
        """Genera conclusiones basadas en los datos"""
        conclusions = []
        
        if len(df['provider'].unique()) < 2:
            conclusions.append("• Solo hay datos de un proveedor. Se necesitan más datos para comparación.")
            return conclusions
        
        # Comparar proveedores
        providers = df['provider'].unique()
        metrics = {}
        
        for provider in providers:
            provider_data = df[df['provider'] == provider]
            metrics[provider] = {
                'avg_latency': provider_data['latency_total'].mean(),
                'avg_cost': provider_data['estimated_cost_usd'].mean(),
                'total_cost': provider_data['estimated_cost_usd'].sum(),
                'avg_tokens': provider_data['estimated_tokens'].mean()
            }
        
        # Hallazgo 1: Costo
        cost_leader = min(providers, key=lambda x: metrics[x]['avg_cost'])
        cost_savings = ((max(metrics[p]['avg_cost'] for p in providers) - metrics[cost_leader]['avg_cost']) / 
                       max(metrics[p]['avg_cost'] for p in providers)) * 100
        conclusions.append(f"• **{cost_leader.upper()}** es {cost_savings:.1f}% más económico por consulta")
        
        # Hallazgo 2: Velocidad
        speed_leader = min(providers, key=lambda x: metrics[x]['avg_latency'])
        speed_improvement = (max(metrics[p]['avg_latency'] for p in providers) / metrics[speed_leader]['avg_latency'])
        conclusions.append(f"• **{speed_leader.upper()}** es {speed_improvement:.1f}x más rápido en respuesta")
        
        # Hallazgo 3: Eficiencia general
        efficiency_scores = {}
        for provider in providers:
            # Puntuación de eficiencia (mayor es mejor)
            efficiency = (1 / metrics[provider]['avg_latency']) * (1 / metrics[provider]['avg_cost'])
            efficiency_scores[provider] = efficiency
        
        efficiency_leader = max(efficiency_scores, key=efficiency_scores.get)
        conclusions.append(f"• **{efficiency_leader.upper()}** ofrece mejor relación costo-rendimiento general")
        
        return conclusions
    
    def _create_html_table(self, comparative_data: List[Dict], conclusions: List[str]) -> str:
        """Crea tabla HTML con estilo profesional"""
        
        # Generar filas de la tabla
        table_rows = ""
        for row in comparative_data:
            table_rows += "<tr>"
            for key, value in row.items():
                table_rows += f"<td>{value}</td>"
            table_rows += "</tr>"
        
        # Generar conclusiones
        conclusions_html = "<br>".join(conclusions)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tabla Comparativa - UFRO Assistant</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .table-container {{ overflow-x: auto; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; background: white; }}
                th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e8f4fd; }}
                .conclusions {{ background-color: #e8f6f3; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .timestamp {{ text-align: right; color: #7f8c8d; font-size: 12px; margin-top: 20px; }}
                .metric-highlight {{ color: #27ae60; font-weight: bold; }}
                .metric-warning {{ color: #e74c3c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Tabla Comparativa - UFRO Assistant</h1>
                
                <h2>Métricas por Proveedor</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                {"".join(f"<th>{key}</th>" for key in comparative_data[0].keys())}
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
                
                <h2>🔍 Hallazgos Principales</h2>
                <div class="conclusions">
                    {conclusions_html}
                </div>
                
                <div class="timestamp">
                    Generado el: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Obtiene consultas recientes"""
        try:
            if not self.metrics_file.exists():
                return []
            
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-limit:]
                return [json.loads(line) for line in lines if line.strip()]
                
        except Exception as e:
            print(f"⚠️ Error obteniendo consultas recientes: {e}")
            return []
    
    def get_provider_stats(self) -> Dict:
        """Obtiene estadísticas de proveedores"""
        try:
            if not self.stats_file.exists():
                return {}
            
            df = pd.read_csv(self.stats_file)
            return df.set_index('provider').to_dict('index')
            
        except Exception as e:
            print(f"⚠️ Error obteniendo estadísticas: {e}")
            return {}

# Instancia global
metrics_logger = MetricsLogger()