FROM python:3.9
WORKDIR /app
COPY preprocess.py iris.csv ./
COPY ../.env .env
RUN pip install pandas psycopg2-binary python-dotenv
CMD ["python", "preprocess.py"]