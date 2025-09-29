# src/eval_flow_best_mlflow.py
import numpy as np
import pandas as pd
import mlflow

from src.base import cost_score

FRAME_CSV = "hex_0.20.csv"
PRED_CSV = "predict.csv"

frame = pd.read_csv(FRAME_CSV)
preds = pd.read_csv(PRED_CSV)

y_true = frame["class"].to_numpy()

if "cal_p2" in preds.columns:
    p_bad = preds["cal_p2"].to_numpy()
elif "p2" in preds.columns:
    p_bad = preds["p2"].to_numpy()
else:
    raise ValueError("Não encontrei coluna de probabilidade ('cal_p2' ou 'p2').")

def eval_at_threshold(thr: float):
    y_hat = np.where(p_bad >= thr, 2, 1)
    acc = float((y_hat == y_true).mean())
    cost = float(cost_score(y_true, y_hat))
    return thr, acc, cost

best_thr, best_acc, best_cost = 0, 0, -1e18
for t in np.linspace(0.0, 1.0, 101):
    thr, acc, cost = eval_at_threshold(t)
    if cost > best_cost:
        best_thr, best_acc, best_cost = thr, acc, cost

# log no MLflow
mlflow.set_experiment("H2O Flow")
with mlflow.start_run(run_name="classification_h2o"):
    mlflow.log_params({"frame_csv": FRAME_CSV, "pred_csv": PRED_CSV})
    mlflow.log_metrics({
        "best_thr":  best_thr,
        "best_cost": best_cost,
        "best_acc":  best_acc,
    })

print(f"Melhor threshold: {best_thr:.2f} | cost={best_cost:.2f}")
