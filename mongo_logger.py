# mongo_logger.py
import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from typing import Optional, Dict, Any
import uuid

class MongoLogger:
    """Logger para MongoDB UFRO Analytics"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoLogger, cls).__new__(cls)
            cls._instance._init_logger()
        return cls._instance
    
    def _init_logger(self):
        """Inicializar conexión a MongoDB"""
        self.logger = logging.getLogger(__name__)
        
        # Obtener URI de MongoDB desde variable de entorno
        self.mongo_uri = os.getenv(
            "MONGO_URI", 
            "mongodb://localhost:27017/ufro_analytics"
        )
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Verificar conexión
            self.client.admin.command('ping')
            self.db = self.client.get_database()
            self.logger.info(f"✅ Conectado a MongoDB: {self.db.name}")
            
            # Crear índices si no existen
            self._ensure_indexes()
            
        except ConnectionFailure as e:
            self.logger.error(f"❌ No se pudo conectar a MongoDB: {e}")
            self.client = None
            self.db = None
    
    def _ensure_indexes(self):
        """Crear índices necesarios para performance"""
        if self.db is None:
            return
        
        # Índices para access_logs
        self.db.access_logs.create_index([("timestamp", -1)])
        self.db.access_logs.create_index([("user.type", 1)])
        self.db.access_logs.create_index([("service_type", 1)])
        self.db.access_logs.create_index([("endpoint", 1)])
        
        # Índice TTL para borrar logs antiguos automáticamente (30 días)
        self.db.access_logs.create_index(
            [("timestamp", 1)], 
            expireAfterSeconds=2592000
        )
        
        self.logger.info("✅ Índices MongoDB creados/verificados")
    
    def is_connected(self):
        """Verificar si hay conexión activa"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False
    
    def log_access(self, access_data: Dict[str, Any]):
        """Guardar un log de acceso en la colección access_logs"""
        if not self.is_connected():
            self.logger.warning("⚠️ MongoDB no conectado, no se guardará log de acceso")
            return
        
        try:
            # Asegurar que tenga timestamp
            if 'timestamp' not in access_data:
                access_data['timestamp'] = datetime.utcnow()
            
            # Insertar en la colección
            result = self.db.access_logs.insert_one(access_data)
            self.logger.debug(f"Log de acceso guardado con id: {result.inserted_id}")
        except Exception as e:
            self.logger.error(f"Error guardando log de acceso: {e}")
    
    def log_service(self, service_data: Dict[str, Any]):
        """Guardar un log de servicio en la colección service_logs"""
        if not self.is_connected():
            self.logger.warning("⚠️ MongoDB no conectado, no se guardará log de servicio")
            return
        
        try:
            # Asegurar que tenga timestamp
            if 'timestamp' not in service_data:
                service_data['timestamp'] = datetime.utcnow()
            
            # Insertar en la colección
            result = self.db.service_logs.insert_one(service_data)
            self.logger.debug(f"Log de servicio guardado con id: {result.inserted_id}")
        except Exception as e:
            self.logger.error(f"Error guardando log de servicio: {e}")
    
    def get_recent_logs(self, limit: int = 10):
        """Obtener logs recientes"""
        if not self.is_connected():
            return []
        
        try:
            return list(self.db.access_logs.find()
                       .sort("timestamp", -1)
                       .limit(limit))
        except Exception as e:
            self.logger.error(f"Error obteniendo logs: {e}")
            return []
    
    def get_metrics_summary(self, days: int = 7):
        """Obtener resumen de métricas"""
        if not self.is_connected():
            return {}
        
        try:
            from datetime import datetime, timedelta
            start_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$service_type",
                    "total_requests": {"$sum": 1},
                    "avg_latency_ms": {"$avg": "$timing_ms"},
                    "success_rate": {
                        "$avg": {
                            "$cond": [{"$lt": ["$status_code", 400]}, 1, 0]
                        }
                    }
                }}
            ]
            
            result = list(self.db.access_logs.aggregate(pipeline))
            return result
        except Exception as e:
            self.logger.error(f"Error obteniendo métricas: {e}")
            return {}

# Instancia global
mongo_logger = MongoLogger()