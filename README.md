# DiabOps
Solution MLOps complète pour la prédiction du diabète : transformation de modèles ML en API de production (FastAPI/Docker). Intègre l'entraînement automatisé (MLflow), un pipeline CI/CD robuste (GitHub Actions) et un monitoring temps réel (Prometheus/Grafana).

```bash
DiabOps/
├── .github/
│   └── workflows/
│       └── ci.yml
│       └── cd.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── raw/
│   │    └── data.csv
│   └── labeled/
│        └── data.csv
├── monitoring/
│   ├── grafana/                
│   │    ├── dashboards/
│   │    │      └── grafana.json
│   │    └── provisioning/
│   │           ├── dashboards/
│   │           │       └── dashboards.yml
│   │           └── datasources/
│   │                   └── prometheus.yml
│   ├── alertmanager.yml
│   ├── alerts.yml
│   └── prometheus.yml
├── notebooks/
│   ├── exploring.ipynb
│   ├── processing.ipynb
│   ├── training.ipynb
│   └── classifying.ipynb
├── src/
│   ├── utils/
│   │    └── scaler.pkl
│   ├── __init__.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_api.py
├── .gitignore
├── .env
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.mlflow
├── README.md
└── requirements.txt
```