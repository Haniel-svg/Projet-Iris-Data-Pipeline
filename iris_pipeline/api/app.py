from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le modèle de données pour la requête
class PredictionRequest(BaseModel):
    sepal_width: float

# Charger le modèle depuis MLflow
try:
    # Connexion à MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Récupérer le dernier run de l'expérience iris_sepal_prediction
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris_sepal_prediction")
    if not experiment:
        raise Exception("Expérience iris_sepal_prediction introuvable")

    # Trouver le dernier run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise Exception("Aucun run trouvé dans l'expérience iris_sepal_prediction")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Modèle chargé avec succès depuis MLflow, run_id: {run_id}")

except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

# Route pour vérifier que l'API fonctionne
@app.get("/")
def read_root():
    return {"message": "API de prédiction Iris est en ligne"}

# Route pour faire des prédictions
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Préparer les données pour la prédiction
        input_data = pd.DataFrame([[request.sepal_width]], columns=["sepal_width"])
        prediction = model.predict(input_data)[0]
        return {"sepal_length_predicted": prediction}
    except Exception as e:
        return {"error": str(e)}