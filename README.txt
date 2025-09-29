Projeto AI
#1 Hipótese - Sklearn normal
#2 Hipótese - H2O Flow
#3 Hipótese - Utilização do Optuna para o tuning de hiperparametros


0) para instalar requirements, usar o ficheiro requirements.txt (pip install -r requirements.txt)

1) EDA (python -m src.eda)

2) Feature engineering - usar features.py para a primeira e terceira hipótese, mas usar o csv_transform_h2o para a segunda - no modelo de classificação (transformar o csv antes de importar o Flow) - e usar csv_transform_h2o_reg no modelo de regressão

3) Para correr o H2O Flow num servidor localhost, usar h2o_init

4) Para correr o mlflow, usar mlflow ui --host 127.0.0.1 --port 5000

