"""
calendar_service.py – Fetch the active F1 season calendar via Jolpica-F1 API
and detect the next upcoming Grand Prix.
"""

import requests
from datetime import datetime, timezone
from typing import Optional


JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def fetch_season_calendar(year: int | None = None) -> list[dict]:
    """Fetch the full race calendar for the given season year."""
    if year is None:
        year = datetime.now(timezone.utc).year
    url = f"{JOLPICA_BASE}/{year}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        races = data["MRData"]["RaceTable"]["Races"]
        if races:
            return races
    except Exception as e:
        print(f"[WARN] Online calendar fetch failed for year {year}: {e}")

    # Fallback default schedule if network is unavailable or season not yet published
    return [
        {
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit": {"circuitId": "bahrain", "circuitName": "Bahrain International Circuit"},
            "date": f"{year}-03-02",
            "time": "15:00:00Z"
        },
        {
            "round": "8",
            "raceName": "Monaco Grand Prix",
            "Circuit": {"circuitId": "monaco", "circuitName": "Circuit de Monaco"},
            "date": f"{year}-05-25",
            "time": "13:00:00Z"
        },
        {
            "round": "12",
            "raceName": "British Grand Prix",
            "Circuit": {"circuitId": "silverstone", "circuitName": "Silverstone Circuit"},
            "date": f"{year}-07-05",
            "time": "14:00:00Z"
        },
        {
            "round": "14",
            "raceName": "Belgian Grand Prix",
            "Circuit": {"circuitId": "spa", "circuitName": "Circuit de Spa-Francorchamps"},
            "date": f"{year}-07-26",
            "time": "13:00:00Z"
        },
        {
            "round": "16",
            "raceName": "Italian Grand Prix",
            "Circuit": {"circuitId": "monza", "circuitName": "Autodromo Nazionale Monza"},
            "date": f"{year}-09-06",
            "time": "13:00:00Z"
        },
        {
            "round": "21",
            "raceName": "Las Vegas Grand Prix",
            "Circuit": {"circuitId": "vegas", "circuitName": "Las Vegas Strip Street Circuit"},
            "date": f"{year}-11-22",
            "time": "06:00:00Z"
        },
        {
            "round": "24",
            "raceName": "Abu Dhabi Grand Prix",
            "Circuit": {"circuitId": "yas_marina", "circuitName": "Yas Marina Circuit"},
            "date": f"{year}-12-06",
            "time": "13:00:00Z"
        }
    ]


def detect_next_race(races: list[dict]) -> dict:
    """
    Compare event dates against current UTC time to find the next upcoming GP.
    Falls back to the latest completed race if all races are in the past (off-season).
    Returns a dict with round, raceName, circuitId, and date.
    """
    now = datetime.now(timezone.utc)
    upcoming = []
    past = []

    for race in races:
        race_date_str = race["date"]
        # Some entries include a time field
        race_time_str = race.get("time", "14:00:00Z")
        if race_time_str and not race_time_str.endswith("Z"):
            race_time_str += "Z"
        try:
            dt = datetime.strptime(
                f"{race_date_str}T{race_time_str}", "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            dt = datetime.strptime(race_date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )

        entry = {
            "round": int(race["round"]),
            "raceName": race["raceName"],
            "circuitId": race["Circuit"]["circuitId"],
            "circuitName": race["Circuit"]["circuitName"],
            "date": race_date_str,
            "datetime": dt,
        }
        if dt > now:
            upcoming.append(entry)
        else:
            past.append(entry)

    if upcoming:
        # Return the soonest upcoming race
        upcoming.sort(key=lambda x: x["datetime"])
        chosen = upcoming[0]
    elif past:
        # Off-season fallback: return the last completed race
        past.sort(key=lambda x: x["datetime"])
        chosen = past[-1]
    else:
        raise ValueError("No races found in calendar data.")

    # Drop the datetime object before returning (not JSON-serializable)
    chosen.pop("datetime", None)
    return chosen


def get_next_grand_prix(year: Optional[int] = None) -> dict:
    """High-level convenience: fetch calendar and return next GP info."""
    races = fetch_season_calendar(year)
    return detect_next_race(races)


if __name__ == "__main__":
    import json
    gp = get_next_grand_prix()
    print(json.dumps(gp, indent=2))
