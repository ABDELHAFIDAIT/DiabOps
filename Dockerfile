# 1. Image de base : Python 3.10 version légère
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /code

# 3. Copier les dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code source de l'API et les modules src
COPY app/ app/
COPY src/ src/

# 5. Copier le dossier MLflow (qui contient le modèle entraîné)
# C'est crucial car on n'a pas de serveur MLflow distant pour ce projet
COPY mlruns/ mlruns/

# 6. Exposer le port 8000
EXPOSE 8000

# 7. Commande de démarrage de l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]