# Guía de Reproducibilidad

## Versiones utilizadas
- Python 3.10+
- Spark 3.5.0
- Pandas 2.0.3

## Pasos para reproducir

1. Clonar repositorio
2. Crear entorno virtual: `python3 -m venv spark_env`
3. Activar: `source spark_env/bin/activate`
4. Instalar: `pip install -r requirements.txt`
5. Ejecutar: `python src/evaluation/experiments.py`

## Seeds para reproducibilidad
Todos los experimentos usan seeds fijos:
- Numpy seed: 42
- Random seed: 42
- PySpark partitions: 4