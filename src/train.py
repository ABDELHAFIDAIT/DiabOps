import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from src.preprocess import create_pipeline

RAW_DATA_PATH = "data/raw/data.csv"
LABELED_DATA_PATH = "data/processed/data.csv"
EXPERIMENT_NAME = "DiabOps_Experiment"
MODEL_NAME = "diabrisk-model"

def train():
    print("Démarrage du script ...")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_labeled = pd.read_csv(LABELED_DATA_PATH)
        print(f"Chargé Raw: {df_raw.shape}, Labelled: {df_labeled.shape}")
    except FileNotFoundError as e:
        print(f"Erreur fichier: {e}")
        return

    drop_cols = ['Cluster', 'risk_category', 'Unnamed: 0']
    
    X_raw = df_raw.drop([c for c in drop_cols if c in df_raw.columns], axis=1)

    X_labeled = df_labeled.drop([c for c in drop_cols if c in df_labeled.columns], axis=1)
    
    if "Cluster" not in df_labeled.columns:
        print("Erreur: Colonne 'Cluster' manquante dans labeled_data.csv")
        return
    
    y_labeled = df_labeled["Cluster"]

    print(f"Features utilisées (Raw): {X_raw.shape}")
    print(f"Features utilisées (Labelled): {X_labeled.shape}")

    X_train_lbl, X_test_lbl, y_train_lbl, y_test_lbl = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=False)

    print("Calibrage du pipeline de nettoyage sur X_raw...")
    preprocessing_pipeline = create_pipeline()
    preprocessing_pipeline.fit(X_raw)

    models_to_train = [
        {
            "name": "SVC",
            "model": SVC(C=10, kernel='linear', gamma='scale', probability=True, random_state=42)
        },
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(C=10, penalty='l1', solver='saga', random_state=42)
        }
    ]

    for item in models_to_train:
        model_name = item["name"]
        clf = item["model"]

        with mlflow.start_run(run_name=model_name) as run:
            print(f"Run ID ({model_name}): {run.info.run_id}")
            print(f"Entraînement du {model_name} sur X_labeled...")
            
            clf.fit(X_train_lbl, y_train_lbl)

            final_model = Pipeline([
                ('preprocessor', preprocessing_pipeline),
                ('classifier', clf)
            ])

            y_pred = clf.predict(X_test_lbl)
            acc = accuracy_score(y_test_lbl, y_pred)
            rec = recall_score(y_test_lbl, y_pred, average='macro')
            f1 = f1_score(y_test_lbl, y_pred, average='macro')

            print(f"Résultats {model_name} :\n- Accuracy: {acc:.4f}\n- Recall: {rec:.4f}\n- F1: {f1:.4f}")
            
            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_recall": rec,
                "test_f1": f1
            })
            mlflow.log_param("model_type", model_name)

            signature = infer_signature(X_raw.head(), clf.predict(X_train_lbl.head()))

            if acc > 0.70:
                print(f"Succès. Enregistrement de {model_name} dans le Registry.")
                mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model", signature=signature)

if __name__ == "__main__":
    train()