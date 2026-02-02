from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from app.schemas import PatientData, PredictionResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow.pyfunc
import mlflow
import joblib
import pandas as pd
import os
import time


model = None
scaler = None

def load_model():
    global model
    global scaler
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        model = mlflow.pyfunc.load_model("models:/DiabRiskModel@champion")
        scaler = joblib.load("./src/utils/scaler.pkl")
        return True
    except Exception:
        return False
    

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    
    yield

    
app = FastAPI(title="DiabRisk API", lifespan=lifespan)

    
@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None, "scaler_loaded": scaler is not None}




# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests",
    ["endpoint"]
)

INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time spent during model inference"
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total number of errors",
    ["endpoint"]
)



@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()

    return response



@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    start = time.time()
    
    if model is None or scaler is None :
        raise HTTPException(status_code=500, detail="Model or scaler not loaded !")
    
    try :
        input_df = pd.DataFrame([data.model_dump()])
        column_names = input_df.columns
        scaled_array = scaler.transform(input_df)
        
        scaled_df = pd.DataFrame(scaled_array, columns=column_names)
        
        prediction = model.predict(scaled_df)
        
        print(prediction)
        
        if hasattr(model, "predict_proba") :
            probability = round(float(app.state.model.predict_proba(scaled_df).max()), 2)
        else :
            probability = None
        
        risk = "High Risk" if prediction[0] == 0 else "Low Risk"
        
        INFERENCE_TIME.observe(time.time() - start)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            risk_level=risk,
            probability=probability
        )
    
    except Exception as e :
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")




@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

