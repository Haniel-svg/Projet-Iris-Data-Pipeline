import os
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_name = "SepalLengthPredictor"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

class InputData(BaseModel):
    sepal_width: float

@app.get("/")
def root():
    return {"message": "API is up!"}

@app.post("/predict")
def predict(data: InputData):
    input_df = {"sepal_width": [data.sepal_width]}
    prediction = model.predict(input_df)
    return {"sepal_length": prediction[0]}
