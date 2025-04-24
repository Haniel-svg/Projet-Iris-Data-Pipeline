import pandas as pd
import os

try:
    print("Chargement de iris.csv...")
    df = pd.read_csv("data/iris.csv")
    print(f"Données chargées : {len(df)} lignes")

    # Sélectionner les colonnes nécessaires
    df = df[["sepal_width", "sepal_length"]]
    print("Colonnes sélectionnées : sepal_width, sepal_length")

    # Sauvegarder dans un CSV
    output_path = "data/iris_processed.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Données sauvegardées dans {output_path}")

except Exception as e:
    print(f"Erreur : {e}")
    raise