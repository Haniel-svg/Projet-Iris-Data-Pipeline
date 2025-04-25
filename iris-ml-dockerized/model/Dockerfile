FROM python:3.9
WORKDIR /app
COPY train_model.py ./
COPY .env .env
COPY artifacts/ artifacts/
RUN pip install pandas scikit-learn matplotlib psycopg2-binary mlflow python-dotenv
CMD ["python", "train_model.py"]
