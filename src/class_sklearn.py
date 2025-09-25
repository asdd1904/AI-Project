# src/class_sklearn_v3.py
# Procura de hiperparâmetros + threshold *dentro* da StratifiedKFold (CV interna).
# Passos:
# 1) Split externo train/test (estratificado).
# 2) Para cada combo de hiperparâmetros:
#    - CV interna (StratifiedKFold):
#      • treina no fold-train, prevê proba no fold-val
#      • escolhe o melhor threshold do fold (maximiza cost_score no fold-val)
#      • guarda o melhor custo do fold e acumula probs/labels para um threshold global
#    - score do combo = média dos melhores custos por fold
#    - threshold_global_combo = melhor threshold nas probs concatenadas das folds
# 3) Refit no train externo com a melhor combo; mede no test externo com threshold_global_combo.

from pathlib import Path
import numpy as np
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from mlflow.models import infer_signature

from src.base import df, preprocessor, cost_score, cv_classification  # cv_classification = StratifiedKFold(...)

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

depth_grid = [6, 10, 14, None]
leaf_grid  = [1, 2, 3]
weight_grid = ["balanced", {1:1, 2:2}]
N_TREES = 800                    # fixo para estabilidade
CALIB_METHOD = "sigmoid"         # Platt (mais estável que isotonic em datasets pequenos)
THRESHOLDS = np.linspace(0.25, 0.60, 36)

best = {
    "cv_mean": -1e18, "cv_std": None, "depth": None, "leaf": None, "w": None,
    "thr_global": 0.5, "pipe_proto": None}

for depth in depth_grid:
    for leaf in leaf_grid:
        for w in weight_grid:
            base_rf = RandomForestClassifier(
                n_estimators=N_TREES,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight=w,
                random_state=42,
                n_jobs=-1
            )
            model = CalibratedClassifierCV(base_rf, method=CALIB_METHOD, cv=3)

            fold_costs = []
            all_p_bad = []
            all_y_val = []

            for tr_idx, va_idx in cv_classification.split(X_train, y_train):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("model", model),
                ])
                pipe.fit(X_tr, y_tr)

                classes = pipe.named_steps["model"].classes_
                idx_bad = list(classes).index(2)
                p_bad_va = pipe.predict_proba(X_va)[:, idx_bad]
                y_va_np = y_va.to_numpy()

                best_fold_cost = -1e18
                for t in THRESHOLDS:
                    yhat = np.where(p_bad_va >= t, 2, 1)
                    c = cost_score(y_va_np, yhat)
                    if c > best_fold_cost:
                        best_fold_cost = c
                fold_costs.append(best_fold_cost)

                all_p_bad.append(p_bad_va)
                all_y_val.append(y_va_np)

            cv_mean = float(np.mean(fold_costs))
            cv_std  = float(np.std(fold_costs))

            all_p_bad = np.concatenate(all_p_bad)
            all_y_val = np.concatenate(all_y_val)
            best_thr_global = THRESHOLDS[0]
            best_thr_cost   = -1e18
            for t in THRESHOLDS:
                yhat = np.where(all_p_bad >= t, 2, 1)
                c = cost_score(all_y_val, yhat)
                if c > best_thr_cost:
                    best_thr_cost, best_thr_global = c, t

            if cv_mean > best["cv_mean"]:
                best.update(
                    cv_mean=cv_mean, cv_std=cv_std, depth=depth, leaf=leaf, w=w,
                    thr_global=float(best_thr_global),
                    pipe_proto=Pipeline([
                        ("prep", preprocessor),
                        ("model", CalibratedClassifierCV(
                            RandomForestClassifier(
                                n_estimators=N_TREES,
                                max_depth=depth,
                                min_samples_leaf=leaf,
                                class_weight=w,
                                random_state=42,
                                n_jobs=-1
                            ),
                            method=CALIB_METHOD, cv=3
                        ))
                    ])
                )

pipe_best = best["pipe_proto"]
pipe_best.fit(X_train, y_train)

classes = pipe_best.named_steps["model"].classes_
idx_bad = list(classes).index(2)
p_bad_test = pipe_best.predict_proba(X_test)[:, idx_bad]
yhat_test = np.where(p_bad_test >= best["thr_global"], 2, 1)

test_cost = float(cost_score(y_test.to_numpy(), yhat_test))
test_acc  = float((yhat_test == y_test.to_numpy()).mean())

mlflow.set_experiment("AI-Classification")
with mlflow.start_run(run_name="classification_model"):
    mlflow.log_params({
        "framework": "sklearn",
        "model": f"RF({N_TREES}) + Calibrated({CALIB_METHOD})",
        "best_max_depth": str(best["depth"]),
        "best_min_samples_leaf": int(best["leaf"]),
        "best_class_weight": str(best["w"]),
        "best_threshold_global": best["thr_global"]
    })
    mlflow.log_metrics({
        "cv_cost_mean": best["cv_mean"],
        "cv_cost_std":  best["cv_std"],
        "test_cost":    test_cost,
        "test_accuracy": test_acc
    })

    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    fig_cm = results_dir / "confmat_cls_sklearn_v3.png"
    ConfusionMatrixDisplay.from_predictions(y_test, yhat_test)
    plt.title(f"CM v3 (t={best['thr_global']:.2f}, depth={best['depth']}, leaf={best['leaf']}, w={best['w']})")
    plt.tight_layout(); plt.savefig(fig_cm); plt.close()
    mlflow.log_artifact(str(fig_cm))

    sig = infer_signature(X_train, pipe_best.predict(X_train))
    mlflow.sklearn.log_model(pipe_best, name="classification-model", signature=sig)

print(
    f"CV mean cost={best['cv_mean']:.2f}±{best['cv_std']:.2f} | "
    f"BEST: depth={best['depth']}, leaf={best['leaf']}, w={best['w']}, thr_global={best['thr_global']:.3f} | "
    f"TEST: cost={test_cost:.2f}, acc={test_acc:.3f}"
)
