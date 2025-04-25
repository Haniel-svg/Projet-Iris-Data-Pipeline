FROM python:3.9
WORKDIR /app
COPY main.py ./
COPY .env .env
RUN pip install fastapi uvicorn mlflow python-dotenv
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]