from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd

app = FastAPI()

class PredictionRequest(BaseModel):
    sepal_width: float

try:
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris_sepal_prediction")
    if not experiment:
        raise Exception("Expérience iris_sepal_prediction introuvable")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise Exception("Aucun run trouvé")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Modèle chargé avec succès, run_id: {run_id}")

except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

@app.get("/")
def read_root():
    return {"message": "API de prédiction Iris est en ligne"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([[request.sepal_width]], columns=["sepal_width"])
        prediction = model.predict(input_data)[0]
        return {"sepal_length_predicted": prediction}
    except Exception as e:
        return {"error": str(e)}