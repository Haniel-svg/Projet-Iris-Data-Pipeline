import pandas as pd
import psycopg2
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)
df = pd.read_sql("SELECT * FROM iris", conn)

X = df[['sepal_width']]
y = df['sepal_length']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("iris-regression")

with mlflow.start_run():
    mlflow.log_param("feature", "sepal_width")
    mlflow.log_metric("r2", model.score(X, y))
    mlflow.sklearn.log_model(model, "model", registered_model_name="SepalLengthPredictor")

    plt.scatter(X, y, color="blue")
    plt.plot(X, y_pred, color="red")
    plt.title("Prediction vs Real")
    plt.savefig("artifacts/plot.png")
    mlflow.log_artifact("artifacts/plot.png")
