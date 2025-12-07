// init-mongo.js
// Script de inicialización para MongoDB UFRO

print('=====================================');
print('🎯 Inicializando MongoDB para UFRO');
print('=====================================');

// Cambiar a la base de datos admin para crear usuario
db = db.getSiblingDB('admin');

// Autenticar con credenciales root
db.auth('admin', 'admin123');

print('✅ Autenticado como admin');

// Crear base de datos si no existe
const dbName = 'ufro_analytics';
db = db.getSiblingDB(dbName);

print(`✅ Usando base de datos: ${dbName}`);

// Crear usuario para la aplicación
try {
    db.createUser({
        user: 'ufro_user',
        pwd: 'ufro_password',
        roles: [
            { role: 'readWrite', db: dbName },
            { role: 'dbAdmin', db: dbName }
        ]
    });
    print('✅ Usuario de aplicación creado: ufro_user / ufro_password');
} catch (e) {
    print(`⚠️  Usuario ya existe: ${e.message}`);
}

// Crear colecciones si no existen
const collections = ['access_logs', 'service_logs', 'users', 'config'];

collections.forEach(collectionName => {
    if (!db.getCollectionNames().includes(collectionName)) {
        db.createCollection(collectionName);
        print(`✅ Colección creada: ${collectionName}`);
    } else {
        print(`⚠️  Colección ya existe: ${collectionName}`);
    }
});

// Crear índices para access_logs
try {
    db.access_logs.createIndex({ "timestamp": -1 });
    db.access_logs.createIndex({ "user.type": 1, "timestamp": -1 });
    db.access_logs.createIndex({ "service_type": 1, "timestamp": -1 });
    db.access_logs.createIndex({ "decision": 1, "timestamp": -1 });
    print('✅ Índices creados para access_logs');
} catch (e) {
    print(`⚠️  Error creando índices: ${e.message}`);
}

// Insertar configuración inicial
try {
    db.config.insertMany([
        {
            "key": "system_name",
            "value": "UFRO Analytics",
            "description": "Nombre del sistema",
            "created_at": new Date()
        },
        {
            "key": "retention_days",
            "value": 30,
            "description": "Días de retención de logs",
            "created_at": new Date()
        }
    ]);
    print('✅ Configuración inicial insertada');
} catch (e) {
    print(`⚠️  Error insertando configuración: ${e.message}`);
}

print('');
print('=====================================');
print('🎉 INICIALIZACIÓN COMPLETADA');
print('=====================================');
print('');
print('📊 Base de datos: ufro_analytics');
print('🔑 Credenciales:');
print('   • Admin:        admin / admin123');
print('   • App User:     ufro_user / ufro_password');
print('   • Mongo Express: admin / express123');
print('');
print('🔗 URI de conexión:');
print('   mongodb://ufro_user:ufro_password@localhost:27017/ufro_analytics');
print('');
print('🌐 Mongo Express: http://localhost:8081');
print('=====================================');