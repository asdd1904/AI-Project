import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/DatasetCredit-g.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    print("\nInfo")
    print(df.info())

    print("\n Primeiras linhas")
    print(df.head())

    print("\n Estatisticas numericas")
    print(df.describe())

    print("\n Estatisticas categoricas")
    print(df.describe(include="object"))

    print("\n Missing values por coluna")
    print(df.isnull().sum())

    if "class" in df.columns:
        class_dist = df["class"].value_counts(normalize=True) * 100
        print("\n Distribuicao do target (class)")
        print(class_dist)

        plt.figure(figsize=(5, 4))
        sns.countplot(x="class", data=df, order=df["class"].value_counts().index)
        plt.title("Distribuicao do target (class)")
        plt.savefig(RESULTS_DIR / "class_distribution.png")
        plt.close()

    if "credit_amount" in df.columns:
        print("\n Estatisticas do credito")
        print(df["credit_amount"].describe())

        plt.figure(figsize=(6, 4))
        sns.histplot(df["credit_amount"], bins=30, kde=True)
        plt.title("Distribuicao do credit_amount")
        plt.savefig(RESULTS_DIR / "credit_amount_hist.png")
        plt.close()

    categorical_cols = ["purpose", "job", "housing"]
    for col in categorical_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True) * 100
            print(f"\n=== Frequencia de {col} ===")
            print(freq)

            plt.figure(figsize=(7, 4))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f"Distribuicao de {col}")
            plt.savefig(RESULTS_DIR / f"{col}_freq.png")
            plt.close()

    if "age" in df.columns:
        invalid_age = df[df["age"] < 18]
        if not invalid_age.empty:
            print(f"\nIdades invalidas (<18): {len(invalid_age)} casos")

    if "num_dependents" in df.columns:
        invalid_dep = df[df["num_dependents"] < 0]
        if not invalid_dep.empty:
            print(f"\nDependentes invalidos (<0): {len(invalid_dep)} casos")


if __name__ == "__main__":
    main()
