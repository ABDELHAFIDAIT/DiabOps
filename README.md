# DiabOps
Solution MLOps complète pour la prédiction du diabète : transformation de modèles ML en API de production (FastAPI/Docker). Intègre l'entraînement automatisé (MLflow), un pipeline CI/CD robuste (GitHub Actions) et un monitoring temps réel (Prometheus/Grafana).

```bash
DiabOps/
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # Pipeline GitHub Actions (Test, Train, Build)
├── app/                       # Le code de l'API (Production)
│   ├── __init__.py
│   ├── main.py                # Point d'entrée FastAPI
│   └── schemas.py             # Modèles Pydantic (Validation des données)
├── data/                      # Stockage local (souvent ignoré par Git)
│   ├── raw/                   # Données brutes (data.csv)
│   └── processed/             # Données nettoyées
├── monitoring/                # Configuration de l'observabilité
│   └── prometheus.yml         # Config de scraping Prometheus
├── notebooks/                 # Vos notebooks existants (Exploration)
│   ├── exploring.ipynb
│   ├── processing.ipynb
│   ├── training.ipynb
│   └── classifying.ipynb
├── src/                       # Le cœur du ML (Refactoring des notebooks)
│   ├── __init__.py
│   ├── preprocess.py          # Pipelines Scikit-learn (Imputation, Scaling)
│   └── train.py               # Script d'entraînement avec MLflow
├── tests/                     # Tests unitaires et d'intégration
│   ├── test_api.py
│   └── test_model.py
├── .gitignore                 # Fichiers à exclure (venv, __pycache__, data/)
├── docker-compose.yml         # Orchestration (API + Prometheus + Grafana)
├── Dockerfile                 # Image pour l'API
├── README.md                  # Documentation du projet
└── requirements.txt           # Dépendances Python
```