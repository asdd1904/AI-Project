# src/reg_rf_sklearn_clean_mlflow.py
# Random Forest (sklearn) — regressão minimal:
# - feature engineering (make_features)
# - CV (cv_regression) compara alvo raw vs log1p -> escolhe o melhor
# - Refit final + TEST
# - MLflow (params + métricas essenciais)
# - Output único no terminal

from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

from src.base import df, within30, cv_regression
from src.features import make_features

SEED = 42
RF_PARAMS = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

X = make_features(df.drop(columns=["credit_amount", "class"]))
y = df["credit_amount"].astype(float)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(exclude=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

def cv_within30(use_log: bool) -> float:
    scores = []
    for tr_idx, va_idx in cv_regression.split(X_tr, y_tr):
        Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        ytr = y_tr.iloc[tr_idx].to_numpy(float)
        yva = y_tr.iloc[va_idx].to_numpy(float)

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(**RF_PARAMS)),
        ])

        if use_log:
            pipe.fit(Xtr, np.log1p(ytr))
            yhat = np.expm1(pipe.predict(Xva))
        else:
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xva)

        scores.append(float(within30(yva, yhat)))
    return float(np.mean(scores))

cv_w30_raw = cv_within30(False)
cv_w30_log = cv_within30(True)
use_log_best = cv_w30_log > cv_w30_raw
cv_w30_best = cv_w30_log if use_log_best else cv_w30_raw


pipe_best = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(**RF_PARAMS)),
])

if use_log_best:
    pipe_best.fit(X_tr, np.log1p(y_tr.to_numpy(float)))
    y_pred_te = np.expm1(pipe_best.predict(X_te))
else:
    pipe_best.fit(X_tr, y_tr)
    y_pred_te = pipe_best.predict(X_te)

test_w30 = float(within30(y_te.to_numpy(float), y_pred_te))
test_mae = float(mean_absolute_error(y_te.to_numpy(float), y_pred_te))

mlflow.set_experiment("AI-Regression")
with mlflow.start_run(run_name="rf_sklearn_clean"):
    mlflow.log_params({
        "framework": "sklearn",
        "model": "RandomForestRegressor",
        **RF_PARAMS,
        "use_log1p_target": use_log_best,
        "cv_within30_raw": cv_w30_raw,
        "cv_within30_log": cv_w30_log,
    })
    mlflow.log_metrics({
        "cv_within30_best": cv_w30_best,
        "test_within30": test_w30,
        "test_MAE": test_mae,
    })
    # guarda o modelo (silencioso)
    mlflow.sklearn.log_model(pipe_best, "rf_sklearn_clean_reg")

print(
    f"CV within30 raw={cv_w30_raw:.3f} | log={cv_w30_log:.3f} "
    f"=> BEST={'log1p' if use_log_best else 'raw'} ({cv_w30_best:.3f}) | "
    f"TEST: within30={test_w30:.3f}, MAE={test_mae:.1f}"
)