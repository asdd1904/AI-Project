import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import h2o
from h2o.estimators import H2ORandomForestEstimator

from src.base import df, cost_score

SEED = 42
N_SPLITS = 5
THRESHOLDS = np.linspace(0.05, 0.95, 37)
rf_params = dict(ntrees=800, max_depth=20, min_rows=2, seed=SEED)

def to_h2o(Xp: pd.DataFrame, yp: pd.Series | None, response="class"):
    dfp = Xp.copy()
    if yp is not None:
        dfp[response] = yp.astype(str)
    hf = h2o.H2OFrame(dfp)
    for col, dt in zip(Xp.columns, Xp.dtypes):
        if dt == "object":
            hf[col] = hf[col].asfactor()
    if yp is not None:
        hf[response] = hf[response].asfactor()
    return hf

def h2o_pred_proba_bad(model, hf):
    pred = model.predict(hf).as_data_frame()
    classes = list(model._model_json["output"]["domains"][-1])  # ['1','2']
    idx_bad = classes.index("2")
    p_cols = [c for c in pred.columns if c.startswith("p")]
    return pred[p_cols].to_numpy()[:, idx_bad]

X = df.drop(columns=["class"])
y = df["class"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                          stratify=y, random_state=SEED)

h2o.init(nthreads=2, max_mem_size="2G")
features = X.columns.tolist()
response = "class"

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pbad = np.zeros(len(X_tr), dtype=float)
oof_y    = y_tr.to_numpy()

for tr_idx, va_idx in cv.split(X_tr, y_tr):
    tr_h = to_h2o(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], response)
    va_h = to_h2o(X_tr.iloc[va_idx], None, response)
    m = H2ORandomForestEstimator(**rf_params)
    m.train(x=features, y=response, training_frame=tr_h)
    oof_pbad[va_idx] = h2o_pred_proba_bad(m, va_h)

oof_y_bin = (oof_y == 2).astype(int)

calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
calibrator.fit(oof_pbad.reshape(-1, 1), oof_y_bin)
oof_pbad_cal = calibrator.predict_proba(oof_pbad.reshape(-1, 1))[:, 1]

fold_tstars, fold_costs = [], []
for tr_idx, va_idx in cv.split(X_tr, y_tr):
    yva = y_tr.iloc[va_idx].to_numpy()
    pva = oof_pbad_cal[va_idx]
    best_c, best_t = -1e18, 0.5
    for t in THRESHOLDS:
        yhat = np.where(pva >= t, 2, 1)
        c = cost_score(yva, yhat)
        if c > best_c:
            best_c, best_t = c, t
    fold_costs.append(best_c)
    fold_tstars.append(best_t)

cv_mean, cv_std = float(np.mean(fold_costs)), float(np.std(fold_costs))
thr_global = float(np.median(fold_tstars))

tr_h = to_h2o(X_tr, y_tr, response)
te_h = to_h2o(X_te, None, response)

rf_final = H2ORandomForestEstimator(**rf_params)
rf_final.train(x=features, y=response, training_frame=tr_h)

pbad_test_raw = h2o_pred_proba_bad(rf_final, te_h)
pbad_test_cal = calibrator.predict_proba(pbad_test_raw.reshape(-1, 1))[:, 1]

yhat_test = np.where(pbad_test_cal >= thr_global, 2, 1)
test_cost = float(cost_score(y_te.to_numpy(), yhat_test))
test_acc  = float((yhat_test == y_te.to_numpy()).mean())

mlflow.set_experiment("H2O Classification")
with mlflow.start_run(run_name="classification_h2o"):
    mlflow.log_params({
        "framework": "h2o",
        "model": "DRF",
        **rf_params,
        "calibration": "Platt(LogReg OOF)",
        "threshold_policy": "median(fold_t*)",
        "thr_range": f"[{THRESHOLDS.min():.2f},{THRESHOLDS.max():.2f}]",
        "seed": SEED,
    })
    mlflow.log_metrics({
        "cv_cost_mean": cv_mean,
        "cv_cost_std":  cv_std,
        "thr_global":   thr_global,
        "test_cost":    test_cost,
        "test_accuracy":test_acc,
    })

print(f"H2O DRF (cal) — CV cost={cv_mean:.2f}±{cv_std:.2f} | "
      f"thr*={thr_global:.2f} | TEST: cost={test_cost:.2f}, acc={test_acc:.3f}")

try:
    h2o.remove_all()
    h2o.cluster().shutdown()
except Exception:
    pass