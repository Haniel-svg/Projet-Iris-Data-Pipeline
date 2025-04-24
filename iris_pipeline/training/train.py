import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Paramètres de connexion à PostgreSQL
db_params = {
    "host": "postgres",
    "database": "iris_db",
    "user": "user",
    "password": "password"
}

try:
    print("Connexion à MLflow...")
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("iris_sepal_prediction")

    print("Connexion à PostgreSQL...")
    conn = psycopg2.connect(**db_params)
    query = "SELECT sepal_width, sepal_length FROM iris"
    df = pd.read_sql(query, conn)
    print(f"Données extraites : {len(df)} lignes")
    conn.close()

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
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)

        # Logger les métriques
        print(f"Log des métriques : RMSE={rmse:.4f}, R2={r2:.4f}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Logger le modèle
        print("Log du modèle dans MLflow...")
        artifact_path = "random_forest_model"
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"Modèle loggé avec succès à l'emplacement : {artifact_path}")

        # Vérifier que l'artefact a été écrit
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"URI de l'artefact : {artifact_uri}")
        if os.path.exists(artifact_uri.replace("file://", "")):
            print("Artefact trouvé sur le système de fichiers")
        else:
            print("Erreur : Artefact non trouvé sur le système de fichiers")

        print(f"Modèle entraîné ! RMSE: {rmse:.4f}, R2: {r2:.4f}")

except Exception as e:
    print(f"Erreur : {e}")
    raise

finally:
    try:
        conn.close()
    except:
        pass