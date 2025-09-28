# src/reg_h2o_clean.py
# H2O Random Forest (DRF) — Regressão clean:
# - feature engineering (make_features)
# - CV com os MESMOS folds (cv_regression) e métrica within30
# - treino final + TEST (within30 e MAE)
# - MLflow (params + métricas essenciais)
# - output único no terminal
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow

import h2o
from h2o.estimators import H2ORandomForestEstimator

from src.base import df, within30, cv_regression
from src.features import make_features

# --------------------------
# Config mínima
# --------------------------
SEED = 42
rf_params = dict(
    ntrees=800,
    max_depth=20,
    min_rows=2,
    seed=SEED,
)

# --------------------------
# Helpers
# --------------------------
def to_h2o_reg(Xp: pd.DataFrame, yp: pd.Series | None, response="credit_amount"):
    """Converte pandas→H2OFrame; marca categóricas como factor; target fica numérico."""
    dfp = Xp.copy()
    if yp is not None:
        dfp[response] = yp.astype(float)
    hf = h2o.H2OFrame(dfp)
    # marcar categóricas como factor
    for col, dt in zip(Xp.columns, Xp.dtypes):
        if dt == "object" or str(dt).startswith("category"):
            hf[col] = hf[col].asfactor()
    return hf

# --------------------------
# Dados + features + split
# --------------------------
X = make_features(df.drop(columns=["credit_amount", "class"]))
y = df["credit_amount"].astype(float)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

# --------------------------
# Init H2O
# --------------------------
h2o.init(nthreads=2, max_mem_size="2G")
features = X.columns.tolist()
response = "credit_amount"

# --------------------------
# 1) CV (within30) com os teus folds
# --------------------------
cv_scores = []
for tr_idx, va_idx in cv_regression.split(X_tr, y_tr):
    Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

    tr_h = to_h2o_reg(Xtr, ytr, response)
    va_hX = to_h2o_reg(Xva, None, response)

    m = H2ORandomForestEstimator(**rf_params)
    m.train(x=features, y=response, training_frame=tr_h)

    yhat_va = m.predict(va_hX).as_data_frame().to_numpy().ravel().astype(float)
    cv_scores.append(float(within30(yva.to_numpy(float), yhat_va)))

cv_mean = float(np.mean(cv_scores))
cv_std  = float(np.std(cv_scores))

# --------------------------
# 2) Treino final + TEST
# --------------------------
tr_h = to_h2o_reg(X_tr, y_tr, response)
te_hX = to_h2o_reg(X_te, None, response)

rf_final = H2ORandomForestEstimator(**rf_params)
rf_final.train(x=features, y=response, training_frame=tr_h)

y_pred_te = rf_final.predict(te_hX).as_data_frame().to_numpy().ravel().astype(float)

test_w30 = float(within30(y_te.to_numpy(float), y_pred_te))
test_mae = float(mean_absolute_error(y_te.to_numpy(float), y_pred_te))

# --------------------------
# 3) MLflow (essencial)
# --------------------------
mlflow.set_experiment("H2O Regression")
with mlflow.start_run(run_name="h2o_drf_clean_reg"):
    mlflow.log_params({
        "framework": "h2o",
        "model": "DRF",
        **rf_params,
        "seed": SEED,
        "features": "make_features",
    })
    mlflow.log_metrics({
        "cv_within30_mean": cv_mean,
        "cv_within30_std":  cv_std,
        "test_within30":    test_w30,
        "test_MAE":         test_mae,
    })

# --------------------------
# Output único
# --------------------------
print(f"H2O DRF — CV within30={cv_mean:.3f}±{cv_std:.3f} | "
      f"TEST: within30={test_w30:.3f}, MAE={test_mae:.1f}")

# --------------------------
# Cleanup
# --------------------------
try:
    h2o.remove_all()
    h2o.cluster().shutdown()
except Exception:
    pass
