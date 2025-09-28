import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.base import df, within30, cv_regression
from src.features import make_features

# 1) Data + FE + split
X = make_features(df.drop(columns=["credit_amount", "class"]))
y = df["credit_amount"].astype(float)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(exclude=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

# 2) Objective
def objective(trial: optuna.Trial) -> float:
    use_log = trial.suggest_categorical("use_log1p", [True, False])

    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 600, 2000, step=200),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model = XGBRegressor(**params)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    scores = []
    for tr_idx, va_idx in cv_regression.split(X_tr, y_tr):
        Xtr, Xva = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        ytr = y_tr.iloc[tr_idx].to_numpy(float)
        yva = y_tr.iloc[va_idx].to_numpy(float)
        if use_log:
            pipe.fit(Xtr, np.log1p(ytr))
            yhat = np.expm1(pipe.predict(Xva))
        else:
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xva)
        scores.append(float(within30(yva, yhat)))

    cv_mean = float(np.mean(scores))
    # pruning (uma etapa só já ajuda)
    trial.report(cv_mean, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return cv_mean

# 3) Study
study = optuna.create_study(direction="maximize", pruner=SuccessiveHalvingPruner())
study.optimize(objective, n_trials=120)

best_params = study.best_params
use_log_best = best_params.pop("use_log1p")

# 4) Refit final + TEST
best_model = XGBRegressor(**best_params)
pipe_best = Pipeline([("prep", preprocessor), ("model", best_model)])

if use_log_best:
    pipe_best.fit(X_tr, np.log1p(y_tr.to_numpy(float)))
    y_pred_te = np.expm1(pipe_best.predict(X_te))
else:
    pipe_best.fit(X_tr, y_tr)
    y_pred_te = pipe_best.predict(X_te)

test_w30 = float(within30(y_te.to_numpy(float), y_pred_te))
test_mae = float(mean_absolute_error(y_te.to_numpy(float), y_pred_te))

# 5) MLflow + artefactos
mlflow.set_experiment("AI-Regression (Optuna)")
with mlflow.start_run(run_name="xgb_optuna_reg"):
    mlflow.log_params({"framework": "sklearn+xgboost", "model": "XGBRegressor",
                       "use_log1p_target": use_log_best, **best_params})
    mlflow.log_metrics({"cv_within30_mean": study.best_value,
                        "test_within30": test_w30, "test_MAE": test_mae})

    results = Path("results"); results.mkdir(exist_ok=True)
    fig_path = results / "reg_xgb_optuna_pred_vs_true.png"
    plt.figure(figsize=(5,5))
    plt.scatter(y_te, y_pred_te, s=10, alpha=0.6)
    lims = [min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("True credit_amount"); plt.ylabel("Predicted credit_amount")
    plt.title(f"XGB+Optuna — Pred vs True — test_w30={test_w30:.3f}, MAE={test_mae:.0f}")
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    mlflow.log_artifact(str(fig_path))

    from mlflow.models import infer_signature
    sig = infer_signature(X_tr, pipe_best.predict(X_tr))
    mlflow.sklearn.log_model(pipe_best, "xgb_optuna_reg", signature=sig)

print(f"[XGB-REG Optuna] best_cv_within30={study.best_value:.3f} | "
      f"params={best_params} | use_log1p={use_log_best} | "
      f"TEST: within30={test_w30:.3f}, MAE={test_mae:.1f}")
