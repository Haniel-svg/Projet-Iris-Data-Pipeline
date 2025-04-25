# 🌸 Iris ML Pipeline - Prédiction de Longueur de Sépale

Ce projet met en place un pipeline Machine Learning complet et **dockerisé** pour prédire la **longueur des sépales** à partir de leur **largeur**, en se basant sur le jeu de données Iris.

---

## 🚀 Fonctionnalités

- 📦 Pipeline ML entièrement conteneurisé avec Docker
- 🧮 Prétraitement et stockage des données dans PostgreSQL
- 🤖 Entraînement d’un modèle de régression linéaire avec MLflow
- 📈 Tracking des expérimentations
- 🌐 API FastAPI pour la prédiction en temps réel

---

## 🧰 Stack Technique

- **Docker / Docker Compose**
- **PostgreSQL**
- **MLflow**
- **scikit-learn**
- **FastAPI**
- **Pandas / psycopg2**

---

## 📂 Arborescence du Projet

```plaintext
.
├── api/
│   └── main.py
├── modeling/
│   └── train_model.py
├── preprocessing/
│   └── preprocessing.py
├── data/
│   └── iris.csv
├── Dockerfile (x3)
├── docker-compose.yml
└── README.md
