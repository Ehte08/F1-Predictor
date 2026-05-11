import pandas as pd
from src.config import ACTIVE_CONSTRUCTORS, ACTIVE_DRIVERS, MIN_YEAR
from src.data.fetcher import load_recent_seasons, load_recent_weather
from src.data.loader import load_ergast, load_season2025, load_weather

# Maps historical team names to current branding
TEAM_ALIASES: dict[str, str] = {
    "Force India": "Aston Martin",
    "Racing Point": "Aston Martin",
    "Alfa Romeo": "Sauber",
    "Lotus F1": "Alpine",      # notebook had a typo 'Apline' here — fixed
    "Renault": "Alpine",
    "Alpine F1 Team": "Alpine",
    "Toro Rosso": "RB F1 Team",
    "AlphaTauri": "RB F1 Team",
}

# Ergast drivers table lacks DOBs for 2025 rookies
MANUAL_DOBS: dict[str, str] = {
    "Bortoleto": "2004-10-14",
    "Hadjar": "2004-09-28",
    "Antonelli": "2006-08-25",
    "Hulkenberg": "1987-08-19",
}

# statusIds that count as a driver-caused retirement
_DRIVER_DNF_IDS = {2, 3, 4, 20, 31, 33, 65, 137, 139, 138}
# statusIds that do NOT count as mechanical/team failure
_NON_TEAM_DNF_IDS = {1, 3, 4, 11, 12, 13, 14, 15, 16, 17, 20, 31, 33, 65, 137, 139, 138}

# Saudi Arabia 2025 had missing grid data in the Ergast export
_SAUDI_2025_GRID = [
    "Verstappen", "Piastri", "Russell", "Leclerc", "Antonelli",
    "Sainz", "Hamilton", "Tsunoda", "Gasly", "Norris",
    "Albon", "Lawson", "Alonso", "Hadjar", "Bearman",
    "Stroll", "Doohan", "Hulkenberg", "Ocon", "Bortoleto",
]
_SAUDI_2025_FINISH = [
    "Piastri", "Verstappen", "Leclerc", "Norris", "Russell",
    "Antonelli", "Hamilton", "Sainz", "Albon", "Hadjar",
    "Alonso", "Lawson", "Bearman", "Ocon", "Hulkenberg",
    "Stroll", "Doohan", "Bortoleto", "Tsunoda", "Gasly",
]

# Weather columns dropped to reduce multicollinearity (kept: rainfall, avg_track_temp, min_humidity)
_DROP_WEATHER_COLS = [
    "avg_air_temp", "max_air_temp", "min_air_temp",
    "avg_humidity", "max_humidity", "max_track_temp", "min_track_temp",
]
_DROP_GEO_COLS = ["driver home", "team home", "country", "location"]


def _nationality(x: str) -> str:
    x = str(x).strip().lower()
    _map = {
        "austrian": "AUT", "austria": "AUT",
        "australian": "AUS", "australia": "AUS",
        "indian": "IND", "india": "IND",
        "indonesian": "INA", "indonesia": "INA",
        "uk": "BRI", "british": "BRI", "england": "BRI", "great britain": "BRI",
        "usa": "AME", "united states": "AME", "american": "AME",
        "fra": "FRE", "france": "FRE", "french": "FRE",
    }
    return _map.get(x, x[:3].upper())


def _patch_saudi_2025(df: pd.DataFrame) -> None:
    """Fill missing Saudi 2025 grid/finish positions in-place."""
    mask = (df["GP name"] == "Saudi Arabian Grand Prix") & (df["date"].dt.year == 2025)
    df.loc[mask, "start"] = df.loc[mask, "driver"].map(
        {d: i + 1 for i, d in enumerate(_SAUDI_2025_GRID)}
    )
    df.loc[mask, "finish"] = df.loc[mask, "driver"].map(
        {d: i + 1 for i, d in enumerate(_SAUDI_2025_FINISH)}
    )


def build_base_dataframe(min_year: int = MIN_YEAR) -> pd.DataFrame:
    """
    Loads, merges, and cleans all F1 data from min_year onward.
    Returns a tidy DataFrame ready for feature engineering.
    """
    raw = load_ergast()

    races = raw["races"].copy()
    races.drop(
        columns=["url", "round", "fp1_date", "fp1_time", "fp2_date", "fp2_time",
                 "fp3_date", "fp3_time", "quali_date", "quali_time",
                 "sprint_date", "sprint_time", "time"],
        inplace=True,
    )
    races.rename(columns={"name": "GP name"}, inplace=True)

    constructors = raw["constructors"].drop(columns=["constructorRef", "url"]).rename(
        columns={"name": "team", "nationality": "team_home"}
    )
    drivers = raw["drivers"].drop(columns=["forename", "driverRef", "code", "number", "url"]).rename(
        columns={"surname": "driver", "nationality": "driver_home"}
    )
    results = raw["results"].drop(
        columns=["positionText", "points", "laps", "time", "milliseconds",
                 "positionOrder", "fastestLapTime", "fastestLapSpeed", "rank", "number"]
    ).rename(columns={"position": "finish", "grid": "start"})
    circuits = raw["circuits"].drop(columns=["circuitRef", "url", "lat", "lng", "alt"]).rename(
        columns={"name": "circuit name"}
    )

    df = (
        races
        .merge(results, on="raceId", how="left")
        .merge(constructors, on="constructorId", how="left")
        .merge(drivers, on="driverId", how="left")
        .merge(circuits, on="circuitId", how="left")
    )

    df = df[df["year"] >= min_year].copy()
    df.drop(columns=["resultId", "constructorId", "fastestLap", "driverId", "circuitId", "raceId"], inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Home-race flags before dropping the nationality columns
    for col in ["driver_home", "team_home", "country"]:
        df[col] = df[col].apply(_nationality)
    df["driver home"] = (df["driver_home"] == df["country"]).astype(int)
    df["team home"] = (df["team_home"] == df["country"]).astype(int)
    df.drop(columns=["driver_home", "team_home"], inplace=True)

    # Append 2025 season (fetched via FastF1)
    season2025 = load_season2025()
    df = pd.concat([df, season2025], ignore_index=True)

    # Append any auto-fetched races beyond the static 2025 cutoff
    recent = load_recent_seasons()
    if not recent.empty:
        df = pd.concat([df, recent], ignore_index=True)

    # DNF classification
    df["driver_dnf"] = df["statusId"].apply(lambda x: 1 if x in _DRIVER_DNF_IDS else 0)
    df["team_dnf"] = df["statusId"].apply(lambda x: 0 if x in _NON_TEAM_DNF_IDS else 1)
    df.drop(columns="statusId", inplace=True)

    # Fill missing DOBs for drivers not in the Ergast export
    for driver, dob_str in MANUAL_DOBS.items():
        df.loc[(df["driver"] == driver) & (df["dob"].isna()), "dob"] = pd.to_datetime(dob_str)

    df["date"] = pd.to_datetime(df["date"])
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    delta_days = pd.to_numeric((df["date"] - df["dob"]).dt.days, errors="coerce")
    df["age_at_race"] = (delta_days / 365.25).fillna(28.0)

    df["team"] = df["team"].replace(TEAM_ALIASES)

    df["driver_active"] = df["driver"].isin(ACTIVE_DRIVERS).astype(int)
    df["team_active"] = df["team"].isin(ACTIVE_CONSTRUCTORS).astype(int)

    # Weather — extend static weather.csv with any auto-fetched race weather
    weather = load_weather()
    recent_wx = load_recent_weather()
    if not recent_wx.empty:
        weather = pd.concat([weather, recent_wx], ignore_index=True).drop_duplicates("date", keep="last")
    df = df.merge(weather, on="date", how="left")

    # Normalise finish: DNF rows → position 20
    df["finish"] = pd.to_numeric(df["finish"].replace("\\N", 20), errors="coerce").fillna(20).astype(int)
    df.loc[((df["team_dnf"] == 1) | (df["driver_dnf"] == 1)) & (df["finish"] != 20), "finish"] = 20

    # One row per driver per race (guards against duplicate DOB merges for shared surnames)
    df = df.drop_duplicates(subset=["driver", "date"])

    _patch_saudi_2025(df)

    df["start"] = pd.to_numeric(df["start"], errors="coerce").fillna(20).astype(int)

    # Manually patch two circuits that never matched via circuitId
    df.loc[df["GP name"] == "Canadian Grand Prix", "circuit name"] = "Circuit Gilles Villeneuve"
    df.loc[df["GP name"] == "Monaco Grand Prix", "circuit name"] = "Circuit de Monaco"

    df.drop(columns=_DROP_WEATHER_COLS + _DROP_GEO_COLS, inplace=True, errors="ignore")
    return df.reset_index(drop=True)
