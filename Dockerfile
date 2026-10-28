
FROM python:3.13-slim

WORKDIR /app

# Instalar una dependencia del sistema que puede necesitar XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar las dependencias de Python
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copiar la API y el modelo entrenado
COPY app.py .
COPY model/ ./model/

# Iniciar el servidor de la API
EXPOSE 80

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
