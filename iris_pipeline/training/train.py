import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
import os
from mlflow.models.signature import infer_signature

try:
       print("Connexion à MLflow...")
       mlflow.set_tracking_uri("http://localhost:5000")
       mlflow.set_experiment("iris_sepal_prediction")

       print("Chargement de iris_processed.csv...")
       df = pd.read_csv("data/iris_processed.csv")
       print(f"Données chargées : {len(df)} lignes")

       # Préparer les données
       X = df[["sepal_width"]]
       y = df["sepal_length"]

       # Entraîner le modèle
       with mlflow.start_run() as run:
           print("Entraînement du modèle...")
           model = RandomForestRegressor(n_estimators=100, random_state=42)
           model.fit(X, y)

           # Prédictions et métriques
           print("Calcul des métriques...")
           y_pred = model.predict(X)
           rmse = root_mean_squared_error(y, y_pred)
           r2 = r2_score(y, y_pred)

           # Logger les métriques
           print(f"Log des métriques : RMSE={rmse:.4f}, R2={r2:.4f}")
           mlflow.log_metric("rmse", rmse)
           mlflow.log_metric("r2", r2)

           # Logger un fichier texte pour tester
           with open("test_artifact.txt", "w") as f:
               f.write("Test d'artefact")
           mlflow.log_artifact("test_artifact.txt")
           print("Fichier test_artifact.txt loggé")

           # Inférer la signature du modèle
           signature = infer_signature(X, y_pred)

           # Logger le modèle dans MLflow
           print("Log du modèle dans MLflow...")
           artifact_path = "random_forest_model"
           mlflow.sklearn.log_model(model, artifact_path, signature=signature)
           print(f"Modèle loggé avec succès à l'emplacement : {artifact_path}")

           # Vérifier que l'artefact a été écrit
           artifact_uri = mlflow.get_artifact_uri(artifact_path)
           print(f"URI de l'artefact : {artifact_uri}")
           artifact_local_path = artifact_uri.replace("file://", "") if artifact_uri.startswith("file://") else artifact_uri
           if os.path.exists(artifact_local_path):
               print("Artefact trouvé sur le système de fichiers")
           else:
               print(f"Erreur : Artefact non trouvé à {artifact_local_path}")

           print(f"Modèle entraîné ! RMSE: {rmse:.4f}, R2: {r2:.4f}")

except Exception as e:
    print(f"Erreur : {e}")
    raise