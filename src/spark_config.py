"""
Configuración de Spark optimizada para Jupyter Lab
"""
import findspark
import os
import logging

# Encontrar instalación de Spark automáticamente
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def crear_sesion_spark(
    app_name="TFM_SeleccionCaracteristicas",
    master="local[8]",
    driver_memory="4g",
    executor_memory="4g",
    ui_port=4040,
    verbose=False
):
    """
    Crea una sesión de Spark configurada para Jupyter Lab
    
    Parámetros:
    -----------
    app_name : str
        Nombre de la aplicación Spark
    master : str
        Dirección del master ('local[*]' para usar todos los cores)
    driver_memory : str
        Memoria del driver (ej: '4g', '8g')
    executor_memory : str
        Memoria de ejecutores
    ui_port : int
        Puerto para Spark Web UI (default: 4040)
    verbose : bool
        Si True, muestra logs detallados
    
    Retorna:
    --------
    SparkSession
        Sesión configurada lista para usar
    
    Ejemplo:
    --------
    >>> spark = crear_sesion_spark("Mi_Experimento")
    >>> print("Web UI: http://localhost:4040")
    """
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "8") \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", str(ui_port)) \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    # Configurar nivel de log
    if verbose:
        spark.sparkContext.setLogLevel("INFO")
    else:
        spark.sparkContext.setLogLevel("WARN")
    
    return spark

def mostrar_info_spark(spark, ui_port=4040):
    """
    Muestra información útil sobre la sesión de Spark
    
    Parámetros:
    -----------
    spark : SparkSession
        Sesión de Spark
    ui_port : int
        Puerto de Spark Web UI
    """
    print("=" * 60)
    print("🔥 INFORMACIÓN DE SPARK SESSION")
    print("=" * 60)
    print(f"✓ App Name: {spark.sparkContext.appName}")
    print(f"✓ Master: {spark.sparkContext.master}")
    print(f"✓ Spark Version: {spark.version}")
    print(f"✓ Python Version: {spark.sparkContext.pythonVer}")
    print(f"✓ Default Parallelism: {spark.sparkContext.defaultParallelism}")
    print(f"✓ Default Min Partitions: {spark.sparkContext.defaultMinPartitions}")
    print("\n🌐 SPARK WEB UI")
    print("=" * 60)
    print(f"📊 Accede en tu navegador a:")
    print(f"   👉 http://localhost:{ui_port}")
    print("\n   Opciones disponibles:")
    print("   - Jobs: Ver trabajos en ejecución/completados")
    print("   - Stages: Ver stages de cada job")
    print("   - Storage: Ver RDDs cacheados")
    print("   - Environment: Configuración de Spark")
    print("   - Executors: Estado de ejecutores")
    print("=" * 60 + "\n")

def parar_spark(spark):
    """Para la sesión de Spark correctamente"""
    if spark:
        spark.stop()
        print("✓ Spark session detenida")