import pandas as pd
import psycopg2
from psycopg2 import sql

# Paramètres de connexion à PostgreSQL
db_params = {
    "host": "postgres",
    "database": "iris_db",
    "user": "user",
    "password": "password"
}

try:
    # Charger iris.csv
    df = pd.read_csv("/app/data/iris.csv")
    print(f"Chargé {len(df)} lignes depuis iris.csv")

    # Se connecter à PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Créer la table iris
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris (
        sepal_length FLOAT,
        sepal_width FLOAT,
        petal_length FLOAT,
        petal_width FLOAT,
        species VARCHAR(50)
    );
    """
    cursor.execute(create_table_query)

    # Vider la table pour éviter les doublons
    cursor.execute("TRUNCATE TABLE iris;")

    # Insérer les données
    for _, row in df.iterrows():
        insert_query = sql.SQL("""
        INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, species)
        VALUES (%s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (
            row['sepal_length'],
            row['sepal_width'],
            row['petal_length'],
            row['petal_width'],
            row['species']
        ))

    # Valider les modifications
    conn.commit()
    print("Données chargées dans PostgreSQL avec succès !")

except Exception as e:
    print(f"Erreur : {e}")
    raise

finally:
    try:
        cursor.close()
        conn.close()
    except:
        pass