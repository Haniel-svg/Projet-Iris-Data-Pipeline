import pandas as pd
import psycopg2
import os

df = pd.read_csv('iris.csv')
df.dropna(inplace=True)

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)

cur = conn.cursor()
cur.execute("DELETE FROM iris;")

for _, row in df.iterrows():
    cur.execute("INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, species) VALUES (%s, %s, %s, %s, %s);",
                tuple(row))

conn.commit()
cur.close()
conn.close()
