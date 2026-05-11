"""
Auto-discovers and fetches completed F1 race results not yet in the local cache.

Uses FastF1's event schedule to find all races after the static Ergast/season2025 cutoff,
fetches results and weather for each, and saves to f1_datasets/season_recent.json.
Called automatically by train.py so the model is always trained on the latest data.
"""
import json
import logging
from datetime import date, timedelta

import fastf1
import pandas as pd

from src.config import CACHE_DIR, DATA_DIR

log = logging.getLogger(__name__)

RECENT_CACHE_PATH = DATA_DIR / "season_recent.json"

# Last race date already covered by Ergast + static season2025.csv
_STATIC_CUTOFF = pd.Timestamp("2025-11-09")

# Wait at least this many days after a race before fetching (results settle)
_RESULT_DELAY_DAYS = 1

# FastF1 team name → internal name used throughout the pipeline
_TEAM_ALIASES: dict[str, str] = {
    "Red Bull Racing": "Red Bull",
    "Aston Martin Aramco": "Aston Martin",
    "Aston Martin F1 Team": "Aston Martin",
    "Aston Martin Aramco F1 Team": "Aston Martin",
    "Alpine F1 Team": "Alpine",
    "Visa Cash App RB": "RB F1 Team",
    "Kick Sauber": "Sauber",
    "Stake F1 Team Kick Sauber": "Sauber",
    "Haas F1 Team": "Haas F1 Team",
    "MoneyGram Haas F1 Team": "Haas F1 Team",
}

# GP EventName → circuit name used in driver_circuit_avg feature
_CIRCUIT_MAP: dict[str, str] = {
    "Bahrain Grand Prix": "Bahrain International Circuit",
    "Saudi Arabian Grand Prix": "Jeddah Corniche Circuit",
    "Australian Grand Prix": "Albert Park Grand Prix Circuit",
    "Japanese Grand Prix": "Suzuka Circuit",
    "Chinese Grand Prix": "Shanghai International Circuit",
    "Miami Grand Prix": "Miami International Autodrome",
    "Emilia Romagna Grand Prix": "Autodromo Enzo e Dino Ferrari",
    "Monaco Grand Prix": "Circuit de Monaco",
    "Spanish Grand Prix": "Circuit de Barcelona-Catalunya",
    "Canadian Grand Prix": "Circuit Gilles Villeneuve",
    "Austrian Grand Prix": "Red Bull Ring",
    "British Grand Prix": "Silverstone Circuit",
    "Belgian Grand Prix": "Circuit de Spa-Francorchamps",
    "Hungarian Grand Prix": "Hungaroring",
    "Dutch Grand Prix": "Circuit Zandvoort",
    "Italian Grand Prix": "Autodromo Nazionale Monza",
    "Azerbaijan Grand Prix": "Baku City Circuit",
    "Singapore Grand Prix": "Marina Bay Street Circuit",
    "United States Grand Prix": "Circuit of the Americas",
    "Mexico City Grand Prix": "Autodromo Hermanos Rodriguez",
    "São Paulo Grand Prix": "Autodromo Jose Carlos Pace",
    "Las Vegas Grand Prix": "Las Vegas Strip Street Circuit",
    "Qatar Grand Prix": "Losail International Circuit",
    "Abu Dhabi Grand Prix": "Yas Marina Circuit",
}


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if RECENT_CACHE_PATH.exists():
        with open(RECENT_CACHE_PATH) as f:
            return json.load(f)
    return {"races": [], "weather": []}


def _save_cache(data: dict) -> None:
    RECENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECENT_CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _cached_keys(data: dict) -> set[str]:
    """'date|GP name' pairs already in the cache."""
    return {f"{r['date']}|{r['GP name']}" for r in data.get("races", [])}


# ── Status mapping ────────────────────────────────────────────────────────────

def _status_to_id(status: str) -> int:
    """
    Map a FastF1 status string to an Ergast-compatible statusId so that
    build_base_dataframe()'s DNF logic (driver_dnf / team_dnf) works correctly.

    statusId=1  → classified finish     → driver_dnf=0, team_dnf=0
    statusId=4  → driver-fault DNF      → driver_dnf=1, team_dnf=0
    statusId=31 → disqualified          → driver_dnf=1, team_dnf=0
    statusId=5  → mechanical/team DNF   → driver_dnf=0, team_dnf=1
    """
    s = (status or "").strip().lower()
    if not s or s == "finished" or s.startswith("+"):
        return 1
    if any(k in s for k in ("accident", "collision", "damage", "spin", "puncture")):
        return 4
    if any(k in s for k in ("disqualif", "excluded", "illegal")):
        return 31
    return 5  # engine, gearbox, hydraulics, etc.


# ── Event discovery ───────────────────────────────────────────────────────────

def get_completed_events(year: int) -> pd.DataFrame:
    """Return all non-testing race events for *year* whose date is safely in the past."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    cutoff = pd.Timestamp(date.today() - timedelta(days=_RESULT_DELAY_DAYS))
    return schedule[pd.to_datetime(schedule["EventDate"]) <= cutoff].reset_index(drop=True)


# ── Single-race fetcher ───────────────────────────────────────────────────────

def _fetch_one_race(
    year: int,
    round_number: int,
    event_name: str,
    event_date: str,
    country: str,
    location: str,
) -> tuple[list[dict], dict] | tuple[None, None]:
    """
    Returns (race_rows, weather_row) for one completed race, or (None, None) on failure.
    race_rows: one dict per driver entry
    weather_row: single dict with date + 3 weather columns
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as e:
        log.warning(f"Could not load session {event_name} {year} R{round_number}: {e}")
        return None, None

    results = session.results
    if results is None or results.empty:
        log.warning(f"Empty results for {event_name} {year}")
        return None, None

    # Summarise weather across the entire race session
    weather_data = session.weather_data
    if weather_data is not None and not weather_data.empty:
        rainfall = int(weather_data["Rainfall"].any())
        avg_track_temp = round(float(weather_data["TrackTemp"].median()), 1)
        min_humidity = round(float(weather_data["Humidity"].min()), 1)
    else:
        rainfall, avg_track_temp, min_humidity = 0, 35.0, 40.0

    weather_row = {
        "date": event_date,
        "rainfall": rainfall,
        "avg_track_temp": avg_track_temp,
        "min_humidity": min_humidity,
    }

    circuit_name = _CIRCUIT_MAP.get(event_name, event_name)
    country_code = country[:3].upper() if country else "UNK"

    race_rows = []
    for _, row in results.iterrows():
        raw_team = str(row.get("TeamName", ""))
        team = _TEAM_ALIASES.get(raw_team, raw_team)

        grid = row.get("GridPosition", None)
        start = int(grid) if pd.notna(grid) and int(grid) != 0 else 20

        pos = row.get("Position", None)
        finish = int(pos) if pd.notna(pos) else 20

        dob_raw = row.get("DateOfBirth", None)
        dob_str = str(pd.Timestamp(dob_raw).date()) if pd.notna(dob_raw) else None

        race_rows.append({
            "year": year,
            "GP name": event_name,
            "date": event_date,
            "start": start,
            "finish": finish,
            "statusId": _status_to_id(str(row.get("Status", "Finished"))),
            "team": team,
            "driver": str(row.get("LastName", "Unknown")),
            "dob": dob_str,
            "circuit name": circuit_name,
            "location": location,
            "country": country_code,
            "driver home": 0,
            "team home": 0,
        })

    return race_rows, weather_row


# ── Main update entry point ───────────────────────────────────────────────────

def update_race_cache(verbose: bool = True) -> int:
    """
    Scan the F1 calendar for the past two years and fetch any completed races
    after the static data cutoff that aren't already cached.

    Saves after each race — safe to interrupt and resume.
    Returns the number of newly fetched races.
    """
    cache = _load_cache()
    cached_keys = _cached_keys(cache)
    fetched = 0

    current_year = date.today().year
    years_to_check = sorted({current_year - 1, current_year})

    for year in years_to_check:
        try:
            events = get_completed_events(year)
        except Exception as e:
            log.warning(f"Could not retrieve {year} schedule: {e}")
            continue

        for _, event in events.iterrows():
            event_ts = pd.Timestamp(event["EventDate"])
            if event_ts <= _STATIC_CUTOFF:
                continue

            event_date = str(event_ts.date())
            event_name = str(event["EventName"])
            cache_key = f"{event_date}|{event_name}"

            if cache_key in cached_keys:
                continue

            if verbose:
                print(f"  Fetching {event_name} {year} ({event_date})...")

            race_rows, weather_row = _fetch_one_race(
                year=year,
                round_number=int(event["RoundNumber"]),
                event_name=event_name,
                event_date=event_date,
                country=str(event.get("Country", "")),
                location=str(event.get("Location", "")),
            )

            if race_rows:
                cache["races"].extend(race_rows)
                cache["weather"].append(weather_row)
                cached_keys.add(cache_key)
                _save_cache(cache)
                fetched += 1
                if verbose:
                    print(f"    ✓ {len(race_rows)} driver rows cached.")
            else:
                if verbose:
                    print(f"    ✗ No data returned.")

    if verbose and fetched == 0:
        print("  Already up to date — no new races to fetch.")

    return fetched


# ── Loaders for cleaner.py ────────────────────────────────────────────────────

def load_recent_seasons() -> pd.DataFrame:
    """Race result rows for all auto-fetched races (empty DataFrame if none yet)."""
    cache = _load_cache()
    rows = cache.get("races", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_recent_weather() -> pd.DataFrame:
    """Weather rows for auto-fetched races, compatible with the static weather.csv merge."""
    cache = _load_cache()
    rows = cache.get("weather", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
