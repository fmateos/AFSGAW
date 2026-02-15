"""
Script de validación del entorno Spark + Jupyter para TFM

Uso:
    python test_spark_setup.py

Este script verifica que:
- Java está instalado
- PySpark funciona
- Spark Web UI está habilitado
- Jupyter Lab está disponible
"""

import sys
import subprocess
from pathlib import Path

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(texto):
    print(f"\n{BLUE}{'='*60}")
    print(f"  {texto}")
    print(f"{'='*60}{RESET}\n")

def print_success(texto):
    print(f"{GREEN}✓ {texto}{RESET}")

def print_error(texto):
    print(f"{RED}✗ {texto}{RESET}")

def print_warning(texto):
    print(f"{YELLOW}⚠ {texto}{RESET}")

def print_info(texto):
    print(f"{BLUE}ℹ {texto}{RESET}")

def check_java():
    """Verifica que Java está instalado"""
    print_header("1. Verificando Java")
    
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Extraer versión de stderr (Java lo imprime allí)
            version_line = result.stderr.split('\n')[0]
            print_success(f"Java instalado: {version_line}")
            return True
        else:
            print_error("Java no responde correctamente")
            return False
    except FileNotFoundError:
        print_error("Java NO está instalado")
        print_info("Instala con: brew install openjdk@17")
        return False
    except Exception as e:
        print_error(f"Error checking Java: {e}")
        return False

def check_python_version():
    """Verifica versión de Python"""
    print_header("2. Verificando Python")
    
    version = sys.version
    print_success(f"Python {version.split()[0]} en {sys.executable}")
    
    if sys.version_info >= (3, 8):
        print_success("Versión compatible ✓")
        return True
    else:
        print_error("Python 3.8+ requerido")
        return False

def check_pyspark():
    """Verifica que PySpark está instalado"""
    print_header("3. Verificando PySpark")
    
    try:
        import pyspark
        print_success(f"PySpark {pyspark.__version__} instalado")
        return True
    except ImportError:
        print_error("PySpark NO está instalado")
        print_info("Instala con: pip install pyspark")
        return False

def check_spark_session():
    """Intenta crear una sesión de Spark"""
    print_header("4. Creando sesión Spark")
    
    try:
        sys.path.insert(0, str(Path.cwd() / 'src'))
        from spark_config import crear_sesion_spark, mostrar_info_spark
        
        print_info("Creando SparkSession...")
        spark = crear_sesion_spark(
            app_name="Test_Setup",
            master="local[*]",
            driver_memory="2g",
            executor_memory="2g",
            ui_port=4040,
            verbose=False
        )
        
        print_success("SparkSession creada ✓")
        mostrar_info_spark(spark, ui_port=4040)
        
        # Test simple
        print_info("Ejecutando test simple...")
        rdd = spark.sparkContext.parallelize(range(1000))
        resultado = rdd.map(lambda x: x**2).sum()
        print_success(f"Test RDD exitoso: suma de cuadrados = {resultado:,}")
        
        spark.stop()
        print_success("Spark session detenida correctamente")
        return True
        
    except Exception as e:
        print_error(f"Error creando SparkSession: {e}")
        print_warning("Verifica que Java está instalado correctamente")
        return False

def check_jupyter():
    """Verifica que Jupyter Lab está instalado"""
    print_header("5. Verificando Jupyter Lab")
    
    try:
        result = subprocess.run(['jupyter', 'lab', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Jupyter Lab {version} instalado")
            return True
        else:
            print_error("Jupyter Lab no responde")
            return False
    except FileNotFoundError:
        print_error("Jupyter Lab NO está instalado")
        print_info("Instala con: pip install jupyterlab")
        return False
    except Exception as e:
        print_error(f"Error checking Jupyter: {e}")
        return False

def check_required_packages():
    """Verifica que los packages esenciales están instalados"""
    print_header("6. Verificando packages esenciales")
    
    packages = {
        'pandas': 'Manipulación de datos',
        'numpy': 'Cálculos numéricos',
        'sklearn': 'Machine learning',
        'matplotlib': 'Visualización',
        'seaborn': 'Visualización estadística',
    }
    
    all_ok = True
    for package, descripcion in packages.items():
        try:
            __import__(package)
            version = __import__(package).__version__
            print_success(f"{package} ({descripcion}): {version}")
        except ImportError:
            print_error(f"{package} NO instalado: {descripcion}")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Verifica que la estructura del proyecto es correcta"""
    print_header("7. Verificando estructura del proyecto")
    
    required_dirs = [
        'src',
        'notebooks',
        'tests',
        'data',
        'results',
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        path = Path.cwd() / dir_name
        if path.exists():
            print_success(f"Directorio '{dir_name}/' existe")
        else:
            print_warning(f"Directorio '{dir_name}/' NO existe (creando...)")
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Directorio '{dir_name}/' creado")
    
    required_files = [
        'src/spark_config.py',
        'requirements.txt',
        'README.md',
    ]
    
    for file_name in required_files:
        path = Path.cwd() / file_name
        if path.exists():
            print_success(f"Archivo '{file_name}' existe")
        else:
            print_warning(f"Archivo '{file_name}' NO existe")
            all_ok = False
    
    return all_ok

def main():
    """Función principal"""
    print(f"\n{BLUE}{'*'*60}")
    print(f"  🔥 VALIDADOR DE ENTORNO SPARK + JUPYTER PARA TFM")
    print(f"{'*'*60}{RESET}")
    
    results = {
        'Java': check_java(),
        'Python': check_python_version(),
        'PySpark': check_pyspark(),
        'Spark Session': check_spark_session(),
        'Jupyter Lab': check_jupyter(),
        'Packages': check_required_packages(),
        'Estructura': check_project_structure(),
    }
    
    # Resumen final
    print_header("RESUMEN FINAL")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"Tests completados: {passed}/{total}\n")
    
    for test_name, result in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"  {test_name}: {status}")
    
    print()
    
    if passed == total:
        print_success("¡ENTORNO CONFIGURADO CORRECTAMENTE! 🎉")
        print_info("Próximo paso: python test_spark_setup.py --launch-jupyter")
        return 0
    else:
        print_error(f"Faltan {total - passed} componente(s)")
        print_info("Revisa los errores arriba y reintenata")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)