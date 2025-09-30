import pandas as pd
from src.base import df
from src.features import make_features

X = make_features(df.drop(columns=["class"]))
y = df["class"]

out = pd.concat([y, X], axis=1)
out.to_csv("data/DatasetCredit-g-feats.csv", index=False)
