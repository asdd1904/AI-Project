import sys, mlflow, optuna, h2o
print("Python:", sys.version.split()[0])
print("MLflow:", mlflow.__version__)
print("Optuna:", optuna.__version__)
h2o.init(nthreads=2, max_mem_size="2G")
h2o.shutdown(prompt=False)
print("H2O ok")
