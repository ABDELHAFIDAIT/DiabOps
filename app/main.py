import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from app.schemas import PatientData

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_name = "diabrisk-model"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"Chargement du modèle depuis : {model_uri} ...")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"Erreur critique au chargement du modèle : {e}")
    
    yield
    
    print("Nettoyage des ressources...")
    model = None



app = FastAPI(
    title="DiabOps API",
    description="API de prédiction du risque de diabète (MLOps Project)",
    version="1.0.0",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)
        
        
        
    

@app.post("/predict")
def predict(patient: PatientData):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")

    try:
        input_data = pd.DataFrame([patient.dict()])
        
        prediction = model.predict(input_data)
        
        risk_label = "Elevé" if prediction[0] == 0 else "Faible"

        return {
            "cluster": int(prediction[0]),
            "risk_prediction": risk_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")