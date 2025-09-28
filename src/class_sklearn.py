# src/class_rf_sklearn_clean_mlflow.py
# Random Forest (sklearn) — classificação minimal:
# - usa preprocessor, folds e métricas do projeto (src.base)
# - calibração + threshold tuning (mediana dos melhores por fold)
# - MLflow com params + métricas essenciais
# - output no terminal: cv_mean, cv_std, thr_global, test_cost, test_acc

from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from src.base import df, preprocessor, cost_score, cv_classification
from src.features import make_features

N_ESTIMATORS = 800
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 3
CALIB_METHOD = "sigmoid"
THRESHOLDS = np.linspace(0.05, 0.95, 37)
SEED = 42

X = make_features(df.drop(columns=["class"]))
y = df["class"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

base_rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    n_jobs=-1,
    random_state=SEED,
)
model = CalibratedClassifierCV(base_rf, method=CALIB_METHOD, cv=3)

fold_costs, fold_tstars = [], []

for tr_idx, va_idx in cv_classification.split(X_tr, y_tr):
    Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(Xtr, ytr)

    idx_bad = list(pipe.named_steps["model"].classes_).index(2)
    p_bad = pipe.predict_proba(Xva)[:, idx_bad]
    yva_np = yva.to_numpy()

    best_c, best_t = -1e18, 0.5
    for t in THRESHOLDS:
        yhat = np.where(p_bad >= t, 2, 1)
        c = cost_score(yva_np, yhat)
        if c > best_c:
            best_c, best_t = c, t

    fold_costs.append(best_c)
    fold_tstars.append(best_t)

cv_mean = float(np.mean(fold_costs))
cv_std = float(np.std(fold_costs))
thr_global = float(np.median(fold_tstars))

pipe_best = Pipeline([("prep", preprocessor), ("model", model)])
pipe_best.fit(X_tr, y_tr)

idx_bad = list(pipe_best.named_steps["model"].classes_).index(2)
p_bad_te = pipe_best.predict_proba(X_te)[:, idx_bad]
yhat_te = np.where(p_bad_te >= thr_global, 2, 1)

test_cost = float(cost_score(y_te.to_numpy(), yhat_te))
test_acc = float((yhat_te == y_te.to_numpy()).mean())

mlflow.set_experiment("AI-Classification")
with mlflow.start_run(run_name="rf_sklearn_clean"):
    mlflow.log_params(
        {
            "framework": "sklearn",
            "model": "RandomForestClassifier+Calibrated",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "calibration": CALIB_METHOD,
            "threshold_policy": "median(fold_t*)",
            "thr_range": f"[{THRESHOLDS.min():.2f},{THRESHOLDS.max():.2f}]",
            "seed": SEED,
            "features": "make_features",  # meramente informativo
        }
    )
    mlflow.log_metrics(
        {
            "cv_cost_mean": cv_mean,
            "cv_cost_std": cv_std,
            "thr_global": thr_global,
            "test_cost": test_cost,
            "test_accuracy": test_acc,
        }
    )
    mlflow.sklearn.log_model(pipe_best, "rf_sklearn_clean")

print(
    f"CV mean cost={cv_mean:.2f}±{cv_std:.2f} | thr*={thr_global:.2f} | "
    f"Test: cost={test_cost:.2f}, acc={test_acc:.3f}"
)