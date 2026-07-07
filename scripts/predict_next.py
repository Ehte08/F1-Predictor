"""
Predict the next upcoming F1 race and publish a locked prediction artifact.

Steps:
  1. find the next race not yet run (FastF1 schedule);
  2. build the grid — real quali classification if available, else a current-form
     placeholder grid from the latest race in the dataset;
  3. fetch a race-day weather forecast from Open-Meteo (free, no key) using the
     circuit lat/lon in f1_datasets/circuits.csv;
  4. generate a locked prediction artifact and set index.json -> next_race.

Idempotent: re-running overwrites the same slug and index entry.

Run:  python scripts/predict_next.py [--n-sims 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.artifacts import compute_rounds, model_version, rebuild_index  # noqa: E402
from src.config import CACHE_DIR, DATA_DIR  # noqa: E402
from src.data.cleaner import build_base_dataframe  # noqa: E402
from src.data.fetcher import _CIRCUIT_MAP, _TEAM_ALIASES  # noqa: E402
from src.site_build import build_eng, predict_upcoming  # noqa: E402


def find_next_event():
    """(EventName, EventDate, RoundNumber, Location, Country) for the next unraced event."""
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    today = pd.Timestamp(date.today())
    for year in (today.year, today.year + 1):
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  schedule {year} unavailable: {e}")
            continue
        upcoming = sched[pd.to_datetime(sched["EventDate"]) > today]
        if not upcoming.empty:
            r = upcoming.iloc[0]
            return (
                str(r["EventName"]),
                pd.Timestamp(r["EventDate"]),
                int(r["RoundNumber"]),
                str(r.get("Location", "")),
                str(r.get("Country", "")),
            )
    return None


def quali_grid(year: int, rnd: int) -> list[dict] | None:
    """Real grid from the quali session, or None if unavailable."""
    import fastf1

    try:
        s = fastf1.get_session(year, rnd, "Q")
        s.load(telemetry=False, weather=False, messages=False)
        res = s.results
        if res is None or res.empty:
            return None
        res = res.sort_values("Position")
        grid = []
        for i, (_, row) in enumerate(res.iterrows(), start=1):
            team = _TEAM_ALIASES.get(str(row.get("TeamName", "")), str(row.get("TeamName", "")))
            grid.append({"driver": str(row.get("LastName", "")), "team": team, "start": i})
        return grid or None
    except Exception as e:
        print(f"  quali unavailable: {e}")
        return None


def placeholder_grid(df: pd.DataFrame) -> list[dict]:
    """Current-form placeholder: latest race's finishing order, driver->team from it."""
    last = df[df["date"] == df["date"].max()].sort_values("finish")
    grid = []
    for i, (_, row) in enumerate(last.iterrows(), start=1):
        grid.append({"driver": str(row["driver"]), "team": str(row["team"]), "start": i})
    return grid


def _circuit_latlon(circuit_name: str):
    try:
        c = pd.read_csv(DATA_DIR / "circuits.csv")
        m = c[c["name"] == circuit_name]
        if not m.empty:
            return float(m.iloc[0]["lat"]), float(m.iloc[0]["lng"])
    except Exception:
        pass
    return None


def fetch_weather(circuit_name: str, race_date: str) -> dict:
    """Open-Meteo daily/hourly forecast → rainfall/avg_track_temp/min_humidity heuristics."""
    default = {"rainfall": 0, "avg_track_temp": 35.0, "min_humidity": 40.0, "source": "manual"}
    latlon = _circuit_latlon(circuit_name)
    if latlon is None:
        return default
    lat, lon = latlon
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max"
        f"&hourly=relative_humidity_2m,temperature_2m"
        f"&start_date={race_date}&end_date={race_date}&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
        daily = data.get("daily", {})
        precip = (daily.get("precipitation_sum") or [0.0])[0] or 0.0
        tmax = (daily.get("temperature_2m_max") or [25.0])[0] or 25.0
        hum = data.get("hourly", {}).get("relative_humidity_2m") or [40.0]
        min_hum = float(min(h for h in hum if h is not None)) if hum else 40.0
        return {
            "rainfall": 1 if float(precip) > 1.0 else 0,
            "avg_track_temp": round(float(tmax) + 12.0, 1),  # track runs hotter than air
            "min_humidity": round(min_hum, 1),
            "source": "open-meteo",
        }
    except Exception as e:
        print(f"  Open-Meteo unavailable ({e}); using manual defaults.")
        return default


def run(n_sims: int = 10_000) -> dict | None:
    version = model_version()
    ev = find_next_event()
    if ev is None:
        print("No upcoming race found.")
        return None
    event_name, event_date, rnd, _location, _country = ev
    race_name = event_name
    circuit = _CIRCUIT_MAP.get(event_name, event_name)
    year = int(event_date.year)
    race_date = event_date.strftime("%Y-%m-%d")
    print(f"Next race: {race_name} ({race_date}), round {rnd}")

    df = build_base_dataframe()
    rounds = compute_rounds(df)
    eng = build_eng(df)

    grid = quali_grid(year, rnd)
    if grid:
        print(f"  Using real quali grid ({len(grid)} drivers).")
    else:
        grid = placeholder_grid(df)
        print(f"  Using current-form placeholder grid ({len(grid)} drivers).")

    weather = fetch_weather(circuit, race_date)
    print(f"  Weather: {weather}")

    slug = predict_upcoming(
        eng, df, rounds,
        race_name=race_name, race_date=race_date, year=year, rnd=rnd,
        circuit=circuit, grid=grid, weather=weather, version=version, n_sims=n_sims,
    )
    rebuild_index(
        version,
        next_race={"slug": slug, "race_name": race_name, "race_date": race_date},
    )
    print(f"Wrote locked prediction → data/site/races/{slug}.json and set index.next_race.")
    return {"slug": slug}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict the next upcoming race.")
    ap.add_argument("--n-sims", type=int, default=10_000)
    args = ap.parse_args()
    run(n_sims=args.n_sims)
