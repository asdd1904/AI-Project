# src/class_xgb_optuna_clean.py
# Optuna + XGBClassifier (sklearn API) — classificação clean:
# - CV com cv_classification + cost_score
# - threshold tuning (por fold, mediana final)
# - refit final + TEST
# - MLflow (essencial) + print único

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from src.base import df, preprocessor, cost_score, cv_classification

SEED = 42
THRESHOLDS = np.linspace(0.05, 0.95, 37)
CALIB_METHOD = "isotonic"

X = df.drop(columns=["class"])
y = df["class"]  # 1=good, 2=bad

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                          stratify=y, random_state=SEED)

def objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 600, 2000, step=200),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )

    base = XGBClassifier(**params)
    model = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=3)

    fold_costs, fold_tstars = [], []
    for tr_idx, va_idx in cv_classification.split(X_tr, y_tr):
        Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(Xtr, (ytr == 2).astype(int))  # 0/1 interno

        idx_bad = list(pipe.named_steps["model"].classes_).index(1)
        p_bad = pipe.predict_proba(Xva)[:, idx_bad]

        best_c, best_t = -1e18, 0.5
        for t in THRESHOLDS:
            yhat = np.where(p_bad >= t, 2, 1)
            c = cost_score(yva.to_numpy(), yhat)
            if c > best_c:
                best_c, best_t = c, t

        fold_costs.append(best_c)
        fold_tstars.append(best_t)

    cv_mean = float(np.mean(fold_costs))
    trial.set_user_attr("tstars", fold_tstars)
    return cv_mean

study = optuna.create_study(direction="maximize", pruner=SuccessiveHalvingPruner())
study.optimize(objective, n_trials=80)

best_params = study.best_params
best_cv_cost = study.best_value
tstars = study.best_trial.user_attrs["tstars"]
thr_global = float(np.median(tstars))

base_best = XGBClassifier(**best_params)
model_best = CalibratedClassifierCV(base_best, method=CALIB_METHOD, cv=3)

pipe_best = Pipeline([("prep", preprocessor), ("model", model_best)])
pipe_best.fit(X_tr, (y_tr == 2).astype(int))

idx_bad = list(pipe_best.named_steps["model"].classes_).index(1)
p_bad_te = pipe_best.predict_proba(X_te)[:, idx_bad]
yhat_te = np.where(p_bad_te >= thr_global, 2, 1)

test_cost = float(cost_score(y_te.to_numpy(), yhat_te))
test_acc = float((yhat_te == y_te.to_numpy()).mean())

mlflow.set_experiment("Optuna Classification")
with mlflow.start_run(run_name="xgb_optuna_cls"):
    mlflow.log_params({
        "framework": "sklearn+xgboost", "model": "XGBClassifier",
        **best_params, "calibration": CALIB_METHOD,
        "thr_policy": "median(fold_t*)", "seed": SEED,
    })
    mlflow.log_metrics({
        "cv_cost_best": best_cv_cost, "thr_global": thr_global,
        "test_cost": test_cost, "test_accuracy": test_acc,
    })
    mlflow.sklearn.log_model(pipe_best, "xgb_optuna_cls")

print(f"[XGB-CLS Optuna] cv_cost={best_cv_cost:.2f} | thr*={thr_global:.2f} | "
      f"TEST: cost={test_cost:.2f}, acc={test_acc:.3f} | params={best_params}")