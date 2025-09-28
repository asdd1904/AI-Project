# src/reg_rf_optuna.py
# RF + Optuna para maximizar within30 em CV e avaliar em TEST (reporta também MAE).
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from pathlib import Path
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.base import df, within30, cv_regression  # usa a mesma métrica e folds :contentReference[oaicite:1]{index=1}

# --------------------------
# Dados & split externo
# --------------------------
X = df.drop(columns=["credit_amount", "class"])
y = df["credit_amount"].astype(float)

# (Usamos o mesmo preprocessor que tens noutro script de regressão)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --------------------------
# Função objetivo Optuna
# --------------------------
def objective(trial: optuna.Trial) -> float:
    n_estimators  = trial.suggest_int("n_estimators", 400, 1400, step=200)
    min_samples   = trial.suggest_int("min_samples_leaf", 1, 6)
    max_depth     = trial.suggest_categorical("max_depth", [None, 12, 16, 20, 24])
    max_features  = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model",
             RandomForestRegressor(
                 n_estimators=n_estimators,
                 min_samples_leaf=min_samples,
                 max_depth=max_depth,
                 max_features=max_features,
                 random_state=42,
                 n_jobs=-1,
             )),
        ]
    )

    # CV: média within30 (igual à definição do src.base)
    scores = []
    for tr_idx, va_idx in cv_regression.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        scores.append(within30(y_va, y_pred))  # mesma função :contentReference[oaicite:2]{index=2}

    cv_mean = float(np.mean(scores))
    trial.report(cv_mean, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return cv_mean  # maximizar within30

# --------------------------
# Estudo Optuna
# --------------------------
study = optuna.create_study(direction="maximize", pruner=SuccessiveHalvingPruner())
study.optimize(objective, n_trials=60, timeout=None)

best_params = study.best_params

# --------------------------
# Refit final + TEST
# --------------------------
best_pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model",
         RandomForestRegressor(
             n_estimators=best_params["n_estimators"],
             min_samples_leaf=best_params["min_samples_leaf"],
             max_depth=best_params["max_depth"],
             max_features=best_params["max_features"],
             random_state=42,
             n_jobs=-1,
         )),
    ]
)
best_pipe.fit(X_train, y_train)
y_pred_test = best_pipe.predict(X_test)

test_within30 = float(within30(y_test, y_pred_test))
test_mae = float(mean_absolute_error(y_test, y_pred_test))

# --------------------------
# MLflow + artefactos
# --------------------------
mlflow.set_experiment("Optuna Regression")
with mlflow.start_run(run_name="reg_optuna"):
    mlflow.log_params({"framework": "sklearn", "model": "RandomForestRegressor", **best_params})
    mlflow.log_metrics({"cv_within30_mean": study.best_value, "test_within30": test_within30, "test_MAE": test_mae})

    # Pred vs True (TEST)
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    fig_path = results_dir / "reg_rf_optuna_pred_vs_true.png"
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred_test, s=8, alpha=0.6)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("True credit_amount"); plt.ylabel("Predicted credit_amount")
    plt.title("RF + Optuna — Predicted vs True (TEST)")
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    mlflow.log_artifact(str(fig_path))

    sig = infer_signature(X_train, best_pipe.predict(X_train))
    mlflow.sklearn.log_model(best_pipe, "rf_optuna_reg", signature=sig)

print(f"[OPTUNA-REG] best_cv_within30={study.best_value:.3f} | params={best_params} | "
      f"TEST: within30={test_within30:.3f}, MAE={test_mae:.1f}")
