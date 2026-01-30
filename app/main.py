import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from app.schemas import PatientData

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    mlflow.set_tracking_uri("file:///code/mlruns")
    experiment_name = "DiabOps_Experiment"
    
    print(f"Recherche du modèle pour : {experiment_name} ...")
    
    try:
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        if current_experiment is None:
            raise Exception("Expérience introuvable dans le conteneur.")

        runs = mlflow.search_runs(
            experiment_ids=[current_experiment.experiment_id],
            order_by=["attribute.start_time DESC"]
        )
        
        if runs.empty:
             raise Exception("Aucun run trouvé.")

        best_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/model"
        
        print(f"Chargement depuis le Run ID : {best_run_id}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modèle chargé avec succès !")

    except Exception as e:
        print(f"Erreur critique : {e}")


        import os
        if os.path.exists("/code/mlruns"):
             print(f"Contenu mlruns : {os.listdir('/code/mlruns')}")
        else:
             print("Dossier /code/mlruns inexistant !")
    
    yield
    model = None



app = FastAPI(title="DiabOps API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)



@app.post("/predict")
def predict(patient: PatientData):
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")
    try:
        input_data = pd.DataFrame([patient.dict()])
        prediction = model.predict(input_data)
        risk_label = "Elevé" if prediction[0] == 0 else "Faible"
        return {"cluster": int(prediction[0]), "risk_prediction": risk_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))