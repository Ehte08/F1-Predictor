import pandas as pd
from src.config import DATA_DIR

_ERGAST_TABLES = ["circuits", "constructors", "drivers", "races", "results"]


def load_ergast() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in _ERGAST_TABLES}


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "weather.csv", index_col=0)
    df["date"] = pd.to_datetime(df["date"])
    # Known one-day offset in source data for Abu Dhabi 2023
    df.loc[df["date"] == pd.Timestamp("2023-11-18"), "date"] = pd.Timestamp("2023-11-19")
    return df


def load_season2025() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "season2025.csv", index_col=0)
