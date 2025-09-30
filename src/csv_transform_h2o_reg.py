import os
import numpy as np
import pandas as pd

USE_EXISTING_FEATS = os.path.exists("data/DatasetCredit-g-feats.csv")

if USE_EXISTING_FEATS:
    df_feats = pd.read_csv("data/DatasetCredit-g-feats.csv")
else:
    from src.base import df            
    from src.features import make_features
    X = make_features(df.drop(columns=["credit_amount", "class"]))
    df_feats = pd.concat([df[["class", "credit_amount"]], X], axis=1)

df_feats["log_credit_amount"] = np.log1p(df_feats["credit_amount"].astype(float))

leak_cols = [c for c in ["credit_per_month", "credit_per_age"] if c in df_feats.columns]
df_out = df_feats.drop(columns=leak_cols)

cols = ["class", "credit_amount", "log_credit_amount"] + \
       [c for c in df_out.columns if c not in ["class", "credit_amount", "log_credit_amount"]]
df_out = df_out[cols]

out_path = "data/DatasetCredit-g-feats-reg-flow.csv"
df_out.to_csv(out_path, index=False)

print(f"OK -> {out_path}")
print(f"Linhas: {len(df_out)} | Features (excl. targets/class): {df_out.shape[1]-3}")
if leak_cols:
    print(f"(Leakage removido {leak_cols})")
else:
    print("(Sem leakage a remover)")
