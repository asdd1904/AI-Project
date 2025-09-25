# src/reg_sklearn_v2.py
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow, mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.base import df, within30_scorer, cv_regression

X = df.drop(columns=["credit_amount", "class"])
y = df["credit_amount"]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numeric_cols     = X.select_dtypes(exclude=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

grid = {
    "n_estimators":      [400, 800, 1200],
    "min_samples_leaf":  [2, 3, 5],
    "max_depth":         [None, 16, 20],
}
combos = list(itertools.product(grid["n_estimators"], grid["min_samples_leaf"], grid["max_depth"]))

results = []
for n_est, leaf, depth in combos:
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=n_est,
            min_samples_leaf=leaf,
            max_depth=depth,
            random_state=42,
            n_jobs=-1
        ))
    ])
    cv_scores = cross_val_score(pipe, X, y, cv=cv_regression, scoring=within30_scorer)
    results.append({
        "n_estimators": n_est,
        "min_samples_leaf": leaf,
        "max_depth": depth,
        "cv_within30_mean": cv_scores.mean(),
        "cv_within30_std":  cv_scores.std()
    })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by=["cv_within30_mean", "cv_within30_std"], ascending=[False, True]).reset_index(drop=True)
best = res_df.iloc[0].to_dict()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

best_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=int(best["n_estimators"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        max_depth=None if pd.isna(best["max_depth"]) else (int(best["max_depth"]) if best["max_depth"] is not None else None),
        random_state=42,
        n_jobs=-1
    ))
])

best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

test_within30 = float(within30_scorer._score_func(y_test, y_pred))
mae = float(mean_absolute_error(y_test, y_pred))

mask = (res_df["min_samples_leaf"] == best["min_samples_leaf"]) & (res_df["max_depth"].astype("object") == best["max_depth"])
line_df = res_df.loc[mask].sort_values("n_estimators")

mlflow.set_experiment("AI-Regression")
with mlflow.start_run(run_name="regression_model"):
    mlflow.log_params({
        "framework": "sklearn",
        "model": "RandomForestRegressor",
        "best_n_estimators": int(best["n_estimators"]),
        "best_min_samples_leaf": int(best["min_samples_leaf"]),
        "best_max_depth": str(best["max_depth"]),
    })
    mlflow.log_metrics({
        "best_cv_within30_mean": float(best["cv_within30_mean"]),
        "best_cv_within30_std":  float(best["cv_within30_std"]),
        "test_within30":         test_within30,
        "test_MAE":              mae
    })
    mlflow.log_artifact(str(csv_path))
    mlflow.log_artifact(str(fig_path))

    sig = infer_signature(X_train, best_pipe.predict(X_train))
    mlflow.sklearn.log_model(best_pipe, name="regression_model", signature=sig)

print(
    f"BEST (CV): n_estimators={int(best['n_estimators'])}, leaf={int(best['min_samples_leaf'])}, depth={best['max_depth']} "
    f"| CV within30={best['cv_within30_mean']:.3f}±{best['cv_within30_std']:.3f}\n"
    f"TEST: within30={test_within30:.3f}, MAE={mae:.1f}"
)
