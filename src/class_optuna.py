from pathlib import Path
import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow.models import infer_signature

from src.base import df, preprocessor, cost_score, cv_classification 

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

THRESHOLDS = np.linspace(0.25, 0.60, 36)

def objective(trial: optuna.Trial) -> float:
    n_estimators = trial.suggest_int("n_estimators", 400, 1200, step=200)
    max_depth = trial.suggest_categorical("max_depth", [6, 10, 14, None])
    min_samples = trial.suggest_int("min_samples_leaf", 1, 5)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", {1: 1, 2: 2}])
    calib_method = trial.suggest_categorical("calibration", ["sigmoid", "isotonic"])

    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(base_rf, method=calib_method, cv=3)

    fold_costs = []
    all_p_bad, all_y_val = [], []

    for tr_idx, va_idx in cv_classification.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_tr, y_tr)

        classes = pipe.named_steps["model"].classes_
        idx_bad = list(classes).index(2)
        p_bad_va = pipe.predict_proba(X_va)[:, idx_bad]
        y_va_np = y_va.to_numpy()

        best_fold = -1e18
        for t in THRESHOLDS:
            yhat = np.where(p_bad_va >= t, 2, 1)
            c = cost_score(y_va_np, yhat)
            if c > best_fold:
                best_fold = c
        fold_costs.append(best_fold)

        all_p_bad.append(p_bad_va)
        all_y_val.append(y_va_np)

    cv_mean = float(np.mean(fold_costs))
    trial.report(cv_mean, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    all_p_bad = np.concatenate(all_p_bad)
    all_y_val = np.concatenate(all_y_val)
    best_thr, best_cost = THRESHOLDS[0], -1e18
    for t in THRESHOLDS:
        yhat = np.where(all_p_bad >= t, 2, 1)
        c = cost_score(all_y_val, yhat)
        if c > best_cost:
            best_cost, best_thr = c, t

    trial.set_user_attr("best_threshold", float(best_thr))
    return cv_mean


study = optuna.create_study(direction="maximize", pruner=SuccessiveHalvingPruner())
study.optimize(objective, n_trials=60, timeout=None)  
best_params = study.best_params
best_thr = study.best_trial.user_attrs.get("best_threshold", 0.5)

base_rf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    class_weight=best_params["class_weight"],
    random_state=42,
    n_jobs=-1,
)
model = CalibratedClassifierCV(base_rf, method=best_params["calibration"], cv=3)
pipe_best = Pipeline([("prep", preprocessor), ("model", model)])
pipe_best.fit(X_train, y_train)

classes = pipe_best.named_steps["model"].classes_
idx_bad = list(classes).index(2)
p_bad_test = pipe_best.predict_proba(X_test)[:, idx_bad]
yhat_test = np.where(p_bad_test >= best_thr, 2, 1)

test_cost = float(cost_score(y_test.to_numpy(), yhat_test))
test_acc = float((yhat_test == y_test.to_numpy()).mean())

mlflow.set_experiment("Optuna Classification")
with mlflow.start_run(run_name="class_optuna"):
    mlflow.log_params({"framework": "sklearn", "model": "RandomForestClassifier", **best_params})
    mlflow.log_metrics({"cv_cost_mean": study.best_value, "test_cost": test_cost, "test_accuracy": test_acc})
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    fig_cm = results_dir / "confmat_cls_rf_optuna.png"
    ConfusionMatrixDisplay.from_predictions(y_test, yhat_test)
    plt.title(f"RF+Optuna — CM (t={best_thr:.2f})")
    plt.tight_layout(); plt.savefig(fig_cm); plt.close()
    mlflow.log_artifact(str(fig_cm))
    sig = infer_signature(X_train, pipe_best.predict(X_train))
    mlflow.sklearn.log_model(pipe_best, "rf_optuna_cls", signature=sig)

print(f"RF_classificação + Optuna - best_cv_cost={study.best_value:.2f} | params={best_params} | t*={best_thr:.3f} | "
      f"Test: cost={test_cost:.2f}, acc={test_acc:.3f}")
