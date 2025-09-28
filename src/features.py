# src/features.py
import pandas as pd
import numpy as np

BINS_AGE = [18, 25, 35, 45, 60, 120]
LABS_AGE = ["18-25", "26-35", "36-45", "46-60", "60+"]

BINS_DURATION = [0, 12, 24, 36, 60, 120, 9999]
LABS_DURATION = ["<=12", "13-24", "25-36", "37-60", "61-120", ">120"]

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    out = a / b2
    return out.fillna(out.median())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece o DF com variáveis comuns em crédito.
    - bins: age, duration
    - ratios: credit_per_month
    - flags: is_rent, no_checking
    - interações: dependents_x_install
    """
    out = df.copy()

    # --- ratios ---
    if {"credit_amount", "duration"}.issubset(out.columns):
        out["credit_per_month"] = _safe_div(out["credit_amount"], out["duration"])

    # --- bins ---
    if "age" in out.columns:
        out["age_bin"] = pd.cut(
            out["age"], bins=BINS_AGE, labels=LABS_AGE,
            include_lowest=True, right=True, ordered=True
        ).astype(str)

    if "duration" in out.columns:
        out["duration_bin"] = pd.cut(
            out["duration"], bins=BINS_DURATION, labels=LABS_DURATION,
            include_lowest=True, right=True, ordered=True
        ).astype(str)

    # --- flags ---
    if "housing" in out.columns:
        out["is_rent"] = (out["housing"].astype(str).str.lower() == "rent").astype(int)

    if "checking_account" in out.columns:
        out["no_checking"] = (out["checking_account"].astype(str).str.lower()
                              .isin(["no_checking", "none", "nan"])).astype(int)

    # --- interações simples ---
    if "num_dependents" in out.columns and "installment_rate" in out.columns:
        out["dependents_x_install"] = out["num_dependents"] * out["installment_rate"]

    # exemplo extra (opcional): carga relativa do crédito
    if {"credit_amount", "age"}.issubset(out.columns):
        out["credit_per_age"] = _safe_div(out["credit_amount"], out["age"].clip(lower=18))

    return out
