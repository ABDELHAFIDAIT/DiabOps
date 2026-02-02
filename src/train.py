import pandas as pd
import numpy as np
from preprocess import preprocess, scaling
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import joblib
import os


RAW_DATA_PATH = "../data/raw/data.csv"
LABELED_DATA_PATH = "../data/labeled/data.csv"
EXPERIMENT_NAME = "Diabetes_Risk_Classification"
MODEL_NAME = "DiabRiskModel"


def train() :
    raw_data = preprocess(RAW_DATA_PATH)
    scaled_data, scaler = scaling(raw_data)
    
    os.makedirs("./utils", exist_ok=True)
    joblib.dump(scaler, "./utils/scaler.pkl")
    
    labeled_data = pd.read_csv(LABELED_DATA_PATH)
    
    X = scaled_data
    y = labeled_data['Cluster']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ros = RandomOverSampler(random_state=42)
    
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    configs = [
        {
            "name" : "SVC",
            "model" : SVC(C=10, gamma='scale', kernel='linear', probability=True, random_state=42),
            "params" : {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
        },
        {
            "name" : "LogisticRegression",
            "model" : LogisticRegression(C=10, penalty='l1', solver='saga', random_state=42),
            "params" : {'C': 10, 'penalty': 'l1', 'solver': 'saga'}
        }
    ]
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    best_run_id = None
    
    for config in configs :
        model_name = config["name"]
        clf = config["model"]
        model_params = config["params"]
        
        with mlflow.start_run(run_name=model_name) as run :
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model_params)
            
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1
            })
            
            signature = infer_signature(X_train, clf.predict(X_train))
            
            mlflow_sklearn.log_model(
                sk_model=clf,
                name="model", 
                signature=signature
            )
            
            mlflow.log_artifact("./utils/scaler.pkl", artifact_path="utils")
            
            if accuracy > best_accuracy :
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = clf
                best_run_id = run.info.run_id
    
    if best_run_id is not None :
        
        model_uri = f"runs:/{best_run_id}/model"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )
        
        client = MlflowClient()
        
        client.set_registered_model_alias(MODEL_NAME, "champion", 1)
        

if __name__ == "__main__" :
    train()
        
    