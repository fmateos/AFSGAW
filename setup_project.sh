#!/bin/bash
set -e

echo "🔧 Configurando entorno TFM..."

# Crear entorno virtual
python3 -m venv spark_env
source spark_env/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Crear estructura de directorios
mkdir -p data/raw data/processed
mkdir -p results/experiments results/metrics results/logs
mkdir -p docs

# Descargar datasets (opcional)
# python src/utils/download_datasets.py

echo "✅ Entorno configurado correctamente"
echo "📌 Para activar el entorno: source spark_env/bin/activate"