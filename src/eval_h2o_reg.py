import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import mlflow

from src.base import within30

FRAME_CSV = "frame_0.20.csv"
PRED_CSV = "pred_regression_v2.csv"

frame = pd.read_csv(FRAME_CSV)
preds = pd.read_csv(PRED_CSV)

if "predict" not in preds.columns:
    raise ValueError(f"Coluna 'predict' não encontrada em {PRED_CSV}. Colunas: {list(preds.columns)}")

y_true = frame["credit_amount"].astype(float).to_numpy()

y_pred_raw = pd.to_numeric(preds["predict"], errors="coerce").to_numpy()
if not np.isfinite(y_pred_raw).all():
    y_pred_raw = np.where(np.isfinite(y_pred_raw), y_pred_raw, np.nan)

if np.isnan(y_true).any() or np.isnan(y_pred_raw).any():
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred_raw)
    y_true = y_true[mask]
    y_pred_raw = y_pred_raw[mask]

mae_raw = float(mean_absolute_error(y_true, y_pred_raw))
w30_raw = float(within30(y_true, y_pred_raw))

SAFE_LIMIT = 700.0
y_pred_log = np.clip(y_pred_raw, -SAFE_LIMIT, SAFE_LIMIT)
y_pred_exp = np.expm1(y_pred_log)

mae_log = float(mean_absolute_error(y_true, y_pred_exp))
w30_log = float(within30(y_true, y_pred_exp))

if mae_log < mae_raw:
    choice = "log1p→expm1"
    mae, w30 = mae_log, w30_log
    thr_note = "(modelo treinado com y=log_credit_amount)"
else:
    choice = "raw"
    mae, w30 = mae_raw, w30_raw
    thr_note = "(modelo treinado com y=credit_amount)"

mlflow.set_experiment("H2O Flow")
with mlflow.start_run(run_name="regression_h2o"):
    mlflow.log_params({
        "frame_csv": FRAME_CSV,
        "pred_csv": PRED_CSV,
        "interpretation": choice,
    })
    mlflow.log_metrics({
        "test_within30_best": w30,
        "test_MAE_best": mae,
        "test_within30_raw": w30_raw,
        "test_MAE_raw": mae_raw,
        "test_within30_log": w30_log,
        "test_MAE_log": mae_log,
    })

print(f"Melhor interpretação: {choice} {thr_note}")
print(f"TEST — within30={w30:.3f} | MAE={mae:.1f}")