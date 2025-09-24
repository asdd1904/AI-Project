import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("AI-Classification")

X, y = load_iris(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with mlflow.start_run(run_name="rf_smoke"):
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    yp = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average="macro")

    mlflow.log_params({"n_estimators": 200, "max_depth": None})
    mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
    mlflow.sklearn.log_model(model, artifact_path="model")

    print({"accuracy": round(acc,4), "f1_macro": round(f1,4)})
