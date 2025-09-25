from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer


df = pd.read_csv("data/DatasetCredit-g.csv")

df["class"] = df["class"].map({"good": 1, "bad": 2})


categorical_cols = df.select_dtypes(include=["object", "category"]).columns
numeric_cols     = df.select_dtypes(exclude=["object", "category"]).columns.drop("class")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

def cost_score(y_true, y_pred):
    total = 0
    for yt, yp in zip(y_true, y_pred):
        if   yt == 1 and yp == 1: total +=   0
        elif yt == 1 and yp == 2: total += -200
        elif yt == 2 and yp == 1: total += -200
        elif yt == 2 and yp == 2: total += +100
    return total / len(y_true)

def within30(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0 
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) < 0.30)

cost_scorer = make_scorer(cost_score, greater_is_better=True)
within30_scorer = make_scorer(within30,  greater_is_better=True)

cv_classification = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_regression     = KFold(        n_splits=5, shuffle=True, random_state=42)
