Dossier Iris_pipeline (ne pas tenir compte des dockerfiles a l'intérieur)
comment tester
python preprocessing/preprocess.py
python training/train.py
lancement mlflow  mlflow ui
lancement api uvicorn main:app --reload
test postman