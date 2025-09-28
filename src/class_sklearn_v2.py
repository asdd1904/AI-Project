# src/class_xgb_sklearn_clean.py
# XGBoost (sklearn API) — classificação clean:
# - y (1/2) -> (0/1) só para treinar; métricas em 1/2
# - sample_weight leve para classe 2 (bad)
# - calibração isotonic + threshold tuning (mediana por fold)
# - MLflow (essencial) + print único

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from src.base import df, preprocessor, cost_score, cv_classification
from src.features import make_features

# ---------------------------
# Config mínima
# ---------------------------
SEED = 42
THRESHOLDS = np.linspace(0.05, 0.95, 37)
CALIB_METHOD = "isotonic"
COST_BAD_WEIGHT = 4.0

xgb_params = dict(
    n_estimators=1600,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.5,
    objective="binary:logistic",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)

# ---------------------------
# Dados
# ---------------------------
X = make_features(df.drop(columns=["class"]))
y12 = df["class"]                 # 1=good, 2=bad (para métricas)
y01 = (y12 == 2).astype(int)      # 0=good, 1=bad (para treinar XGB/calibração)

X_tr, X_te, y_tr12, y_te12, y_tr01, y_te01 = train_test_split(
    X, y12, y01, test_size=0.20, stratify=y12, random_state=SEED
)

# ---------------------------
# Modelo + calibração
# ---------------------------
base = XGBClassifier(**xgb_params)
model = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=3)

# ---------------------------
# CV + threshold tuning
# ---------------------------
fold_costs, fold_tstars = [], []

for tr_idx, va_idx in cv_classification.split(X_tr, y_tr12):
    Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    ytr12 = y_tr12.iloc[tr_idx].to_numpy()
    ytr01 = y_tr01.iloc[tr_idx].to_numpy()
    yva12 = y_tr12.iloc[va_idx].to_numpy()

    # pesos cost-sensitive (classe 2 = bad)
    w = np.where(ytr12 == 2, COST_BAD_WEIGHT, 1.0)

    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(Xtr, ytr01, model__sample_weight=w)

    # prob da classe 1 (bad) no espaço 0/1
    idx_bad = list(pipe.named_steps["model"].classes_).index(1)
    p_bad = pipe.predict_proba(Xva)[:, idx_bad]

    # melhor threshold do fold em labels 1/2
    best_c, best_t = -1e18, 0.5
    for t in THRESHOLDS:
        yhat12 = np.where(p_bad >= t, 2, 1)
        c = cost_score(yva12, yhat12)
        if c > best_c:
            best_c, best_t = c, t

    fold_costs.append(best_c)
    fold_tstars.append(best_t)

cv_mean, cv_std = float(np.mean(fold_costs)), float(np.std(fold_costs))
thr_global = float(np.median(fold_tstars))

# ---------------------------
# Refit final + TEST
# ---------------------------
pipe_best = Pipeline([("prep", preprocessor), ("model", model)])
w_full = np.where(y_tr12.to_numpy() == 2, COST_BAD_WEIGHT, 1.0)
pipe_best.fit(X_tr, y_tr01, model__sample_weight=w_full)

idx_bad = list(pipe_best.named_steps["model"].classes_).index(1)
p_bad_te = pipe_best.predict_proba(X_te)[:, idx_bad]
yhat_te12 = np.where(p_bad_te >= thr_global, 2, 1)

test_cost = float(cost_score(y_te12.to_numpy(), yhat_te12))
test_acc  = float((yhat_te12 == y_te12.to_numpy()).mean())

# ---------------------------
# MLflow (essencial)
# ---------------------------
mlflow.set_experiment("AI-Classification")
with mlflow.start_run(run_name="xgb_sklearn_clean_costsens_iso"):
    mlflow.log_params({
        "framework": "sklearn+xgboost",
        "model": "XGBClassifier",
        **xgb_params,
        "calibration": CALIB_METHOD,
        "threshold_policy": "median(fold_t*)",
        "labels": "train=0/1, metrics=1/2",
        "COST_BAD_WEIGHT": COST_BAD_WEIGHT,
        "seed": SEED,
    })
    mlflow.log_metrics({
        "cv_cost_mean": cv_mean,
        "cv_cost_std":  cv_std,
        "thr_global":   thr_global,
        "test_cost":    test_cost,
        "test_accuracy":test_acc,
    })
    mlflow.sklearn.log_model(pipe_best, "xgb_sklearn_clean_cls")

# ---------------------------
# Output único
# ---------------------------
print(f"CV cost={cv_mean:.2f}±{cv_std:.2f} | thr*={thr_global:.2f} | "
      f"TEST: cost={test_cost:.2f}, acc={test_acc:.3f} | w_bad={COST_BAD_WEIGHT}")

