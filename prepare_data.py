import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PAD = "data/synthetic_news_data.csv"


def load_data(pad=DATA_PAD):
    
    return pd.read_csv(pad)


def show(df):
    
    print(df.shape)
    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())
    print(df.describe())
    print(df["died"].value_counts())


def check_data(df):

    print(df["sup"].value_counts())
    print(df["alert"].value_counts())
    print(df[df["dias"] >= df["syst"]])
    print(df[df["dias"] < 30])
    print(df.groupby("died")[["age", "NEWS", "pulse", "resp", "sat", "temp", "syst"]].mean())
    print(df.groupby("NEWS")["died"].mean())

GRENZEN = {
    "age": (0, 120),
    "syst": (50, 260),
    "dias": (10, 150),
    "temp": (30, 45),
    "pulse": (20, 250),
    "resp": (4, 60),
    "sat": (50, 100),
}


def find_impossible_value(df, grenzen=GRENZEN):
    
    buiten = pd.Series(False, index=df.index)
    for kolom, (laag, hoog) in grenzen.items():
        buiten |= (df[kolom] < laag) | (df[kolom] > hoog)
    return df[buiten]


def clean_data(df):
    
    df = df.copy()
    df["temp"] = df["temp"].round(1)
    onmogelijk = find_impossible_value(df)
    return df.drop(index=onmogelijk.index)


if __name__ == "__main__":
    df = load_data()
    show(df)
    check_data(df)
    print(find_impossible_value(df))
    df = clean_data(df)
    print(df.shape)