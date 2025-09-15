# Étape 1 : base image avec Python
FROM python:3.11-slim

# Crée un dossier de travail
WORKDIR /app

# Copie tout le contenu local dans le conteneur

# Crée les dossiers nécessaires (ex: logs)
RUN mkdir -p api/logs

# Installe les dépendances Python
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Expose le port FastAPI
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
