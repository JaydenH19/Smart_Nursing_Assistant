import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PAD = "data/synthetic_news_data.csv"


def load_data(pad=DATA_PAD):
    """Laadt de dataset en geeft een DataFrame terug."""
    return pd.read_csv(pad)


def show(df):
    """Print een overzicht om de data te leren kennen."""
    print(df.shape)
    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())
    print(df.describe())
    print(df["died"].value_counts())


def check_data(df):
    """Print controles om de kolommen en het label te begrijpen."""
    print(df["sup"].value_counts())
    print(df["alert"].value_counts())
    print(df[df["dias"] >= df["syst"]])
    print(df[df["dias"] < 30])
    print(df.groupby("died")[["age", "NEWS", "pulse", "resp", "sat", "temp", "syst"]].mean())
    print(df.groupby("NEWS")["died"].mean())


if __name__ == "__main__":
    df = load_data()
    show(df)
    check_data(df)