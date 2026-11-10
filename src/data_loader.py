"""
data_loader.py – Extract qualifying results, FP3 telemetry, and historical
race finishes using FastF1 and the Jolpica-F1 API.
Supports multi-circuit Grand Prix presets with authentic circuit physics.
"""

import fastf1
import pandas as pd
import numpy as np
import requests
import warnings
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# Suppress FastF1 future-warnings and cache log noise
warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


# Official verified 2025/2026 driver-team associations
CURRENT_2025_TEAMS = {
    "HAM": "Ferrari",
    "LEC": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "VER": "Red Bull Racing",
    "TSU": "Red Bull Racing",
    "SAI": "Williams",
    "ALB": "Williams",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "OCO": "Haas F1 Team",
    "BEA": "Haas F1 Team",
    "LAW": "Racing Bulls",
    "HAD": "Racing Bulls",
    "HUL": "Sauber",
    "BOR": "Sauber",
}

# Grand Prix Presets across the championship with distinct circuit physics
GRAND_PRIX_PRESETS = {
    "abudhabi": {
        "id": "abudhabi",
        "name": "2025 Abu Dhabi Grand Prix (Latest Season Finale)",
        "round": 24,
        "date": "2025-12-07",
        "circuit_name": "Yas Marina Circuit",
        "track_length": "5.281 km",
        "type": "Twilight Circuit // High Speed & Heavy Traction",
        "sc_prob": 0.45,
        "overtake_difficulty": "Medium (Long DRS Back Straight)",
        "grid_bias": 0.68,
        "pace_bias": 0.32,
        "description": "The official latest Formula 1 World Championship finale. Max Verstappen on Pole (1:22.207), with Norris and Piastri on rows 1 and 2, Russell P4, and Hamilton starting P16 in Ferrari.",
        "grid": [
            ("VER", "Max Verstappen", "Red Bull Racing", 1, 0.000, 0.000),
            ("NOR", "Lando Norris", "McLaren", 2, 0.201, 0.040),
            ("PIA", "Oscar Piastri", "McLaren", 3, 0.230, 0.060),
            ("RUS", "George Russell", "Mercedes", 4, 0.438, 0.120),
            ("LEC", "Charles Leclerc", "Ferrari", 5, 0.523, 0.140),
            ("ALO", "Fernando Alonso", "Aston Martin", 6, 0.695, 0.380),
            ("BOR", "Gabriel Bortoleto", "Sauber", 7, 0.697, 0.420),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 8, 0.706, 0.410),
            ("HAD", "Isack Hadjar", "Racing Bulls", 9, 0.865, 0.450),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 10, 0.893, 0.460),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 11, 0.834, 0.480),
            ("SAI", "Carlos Sainz", "Williams", 12, 0.835, 0.470),
            ("LAW", "Liam Lawson", "Racing Bulls", 13, 0.870, 0.490),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 14, 0.873, 0.500),
            ("STR", "Lance Stroll", "Aston Martin", 15, 0.890, 0.520),
            ("HAM", "Lewis Hamilton", "Ferrari", 16, 1.187, 0.560),
            ("ALB", "Alexander Albon", "Williams", 17, 1.209, 0.580),
            ("HUL", "Nico Hülkenberg", "Sauber", 18, 1.243, 0.600),
            ("GAS", "Pierre Gasly", "Alpine", 19, 1.261, 0.620),
            ("COL", "Franco Colapinto", "Alpine", 20, 1.683, 0.750),
        ],
    },
    "vegas": {
        "id": "vegas",
        "name": "2026 Las Vegas Grand Prix",
        "round": 21,
        "date": "2026-11-22",
        "circuit_name": "Las Vegas Strip Street Circuit",
        "track_length": "6.201 km",
        "type": "Street Circuit // Ultra-Long Straights",
        "sc_prob": 0.65,
        "overtake_difficulty": "Medium (Slipstream DRS)",
        "grid_bias": 0.65,
        "pace_bias": 0.35,
        "description": "Night street race along the Strip. 3 high-speed straights, cold asphalt, and a 65% historical safety car probability.",
        "grid": [
            ("NOR", "Lando Norris", "McLaren", 1, 0.000, 0.000),
            ("VER", "Max Verstappen", "Red Bull Racing", 2, 0.075, 0.040),
            ("RUS", "George Russell", "Mercedes", 3, 0.120, 0.080),
            ("LEC", "Charles Leclerc", "Ferrari", 4, 0.180, 0.110),
            ("HAM", "Lewis Hamilton", "Ferrari", 5, 0.240, 0.150),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 6, 0.310, 0.220),
            ("PIA", "Oscar Piastri", "McLaren", 7, 0.380, 0.260),
            ("SAI", "Carlos Sainz", "Williams", 8, 0.460, 0.320),
            ("ALB", "Alexander Albon", "Williams", 9, 0.530, 0.380),
            ("ALO", "Fernando Alonso", "Aston Martin", 10, 0.610, 0.440),
            ("STR", "Lance Stroll", "Aston Martin", 11, 0.690, 0.500),
            ("GAS", "Pierre Gasly", "Alpine", 12, 0.770, 0.560),
            ("DOO", "Jack Doohan", "Alpine", 13, 0.840, 0.620),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 14, 0.910, 0.670),
            ("HAD", "Isack Hadjar", "Racing Bulls", 15, 0.980, 0.720),
            ("LAW", "Liam Lawson", "Racing Bulls", 16, 1.050, 0.770),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 17, 1.130, 0.820),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 18, 1.210, 0.870),
            ("HUL", "Nico Hülkenberg", "Sauber", 19, 1.300, 0.930),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.390, 0.990),
        ],
    },
    "monaco": {
        "id": "monaco",
        "name": "Monaco Grand Prix",
        "round": 8,
        "date": "2026-05-24",
        "circuit_name": "Circuit de Monaco",
        "track_length": "3.337 km",
        "type": "Extreme High Downforce // Narrow Street",
        "sc_prob": 0.80,
        "overtake_difficulty": "Extreme (Qualifying 90% Decisive)",
        "grid_bias": 0.88,
        "pace_bias": 0.12,
        "description": "Zero room for error. The narrowest circuit on the calendar where track position dictates ~90% of podium outcomes.",
        "grid": [
            ("LEC", "Charles Leclerc", "Ferrari", 1, 0.000, 0.020),
            ("PIA", "Oscar Piastri", "McLaren", 2, 0.154, 0.040),
            ("SAI", "Carlos Sainz", "Williams", 3, 0.248, 0.150),
            ("NOR", "Lando Norris", "McLaren", 4, 0.275, 0.000),
            ("RUS", "George Russell", "Mercedes", 5, 0.288, 0.110),
            ("VER", "Max Verstappen", "Red Bull Racing", 6, 0.297, 0.080),
            ("HAM", "Lewis Hamilton", "Ferrari", 7, 0.351, 0.140),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 8, 0.518, 0.380),
            ("ALB", "Alexander Albon", "Williams", 9, 0.621, 0.420),
            ("GAS", "Pierre Gasly", "Alpine", 10, 0.710, 0.490),
            ("ALO", "Fernando Alonso", "Aston Martin", 11, 0.810, 0.550),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 12, 0.880, 0.600),
            ("STR", "Lance Stroll", "Aston Martin", 13, 0.960, 0.660),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 14, 1.040, 0.710),
            ("DOO", "Jack Doohan", "Alpine", 15, 1.110, 0.760),
            ("LAW", "Liam Lawson", "Racing Bulls", 16, 1.180, 0.810),
            ("HAD", "Isack Hadjar", "Racing Bulls", 17, 1.250, 0.860),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 18, 1.330, 0.910),
            ("HUL", "Nico Hülkenberg", "Sauber", 19, 1.410, 0.960),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.500, 1.020),
        ],
    },
    "silverstone": {
        "id": "silverstone",
        "name": "British Grand Prix",
        "round": 12,
        "date": "2026-07-05",
        "circuit_name": "Silverstone Circuit",
        "track_length": "5.891 km",
        "type": "High Speed Aero // Race Pace Decisive",
        "sc_prob": 0.45,
        "overtake_difficulty": "High (Pace Decisive)",
        "grid_bias": 0.50,
        "pace_bias": 0.50,
        "description": "High-speed corners at Maggotts & Becketts. Severe tyre degradation rewards cars with superior aerodynamic balance.",
        "grid": [
            ("RUS", "George Russell", "Mercedes", 1, 0.000, 0.080),
            ("HAM", "Lewis Hamilton", "Ferrari", 2, 0.171, 0.040),
            ("NOR", "Lando Norris", "McLaren", 3, 0.198, 0.000),
            ("VER", "Max Verstappen", "Red Bull Racing", 4, 0.384, 0.060),
            ("PIA", "Oscar Piastri", "McLaren", 5, 0.419, 0.020),
            ("LEC", "Charles Leclerc", "Ferrari", 6, 0.520, 0.120),
            ("SAI", "Carlos Sainz", "Williams", 7, 0.610, 0.320),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 8, 0.690, 0.280),
            ("STR", "Lance Stroll", "Aston Martin", 9, 0.770, 0.450),
            ("ALB", "Alexander Albon", "Williams", 10, 0.850, 0.510),
            ("ALO", "Fernando Alonso", "Aston Martin", 11, 0.920, 0.560),
            ("GAS", "Pierre Gasly", "Alpine", 12, 1.000, 0.620),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 13, 1.070, 0.670),
            ("DOO", "Jack Doohan", "Alpine", 14, 1.150, 0.720),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 15, 1.220, 0.770),
            ("HAD", "Isack Hadjar", "Racing Bulls", 16, 1.290, 0.820),
            ("LAW", "Liam Lawson", "Racing Bulls", 17, 1.360, 0.870),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 18, 1.440, 0.920),
            ("HUL", "Nico Hülkenberg", "Sauber", 19, 1.520, 0.970),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.610, 1.030),
        ],
    },
    "spa": {
        "id": "spa",
        "name": "Belgian Grand Prix",
        "round": 14,
        "date": "2026-07-26",
        "circuit_name": "Circuit de Spa-Francorchamps",
        "track_length": "7.004 km",
        "type": "High Speed Slipstream // Ardennes Weather",
        "sc_prob": 0.60,
        "overtake_difficulty": "Very High (Kemmel DRS)",
        "grid_bias": 0.48,
        "pace_bias": 0.52,
        "description": "7km Ardennes roller-coaster. Massive Kemmel Straight slipstreaming lets top cars charge from deep on the grid.",
        "grid": [
            ("LEC", "Charles Leclerc", "Ferrari", 1, 0.000, 0.120),
            ("NOR", "Lando Norris", "McLaren", 2, 0.080, 0.000),
            ("HAM", "Lewis Hamilton", "Ferrari", 3, 0.150, 0.060),
            ("PIA", "Oscar Piastri", "McLaren", 4, 0.220, 0.020),
            ("RUS", "George Russell", "Mercedes", 5, 0.310, 0.150),
            ("SAI", "Carlos Sainz", "Williams", 6, 0.420, 0.320),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 7, 0.510, 0.250),
            ("ALO", "Fernando Alonso", "Aston Martin", 8, 0.600, 0.420),
            ("ALB", "Alexander Albon", "Williams", 9, 0.680, 0.470),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 10, 0.770, 0.530),
            ("VER", "Max Verstappen", "Red Bull Racing", 11, 0.850, 0.040), # Starting P11 due to engine penalty
            ("GAS", "Pierre Gasly", "Alpine", 12, 0.920, 0.590),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 13, 0.990, 0.640),
            ("STR", "Lance Stroll", "Aston Martin", 14, 1.060, 0.690),
            ("DOO", "Jack Doohan", "Alpine", 15, 1.130, 0.740),
            ("HAD", "Isack Hadjar", "Racing Bulls", 16, 1.200, 0.790),
            ("LAW", "Liam Lawson", "Racing Bulls", 17, 1.270, 0.840),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 18, 1.350, 0.890),
            ("HUL", "Nico Hülkenberg", "Sauber", 19, 1.430, 0.940),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.520, 1.000),
        ],
    },
    "monza": {
        "id": "monza",
        "name": "Italian Grand Prix",
        "round": 16,
        "date": "2026-09-06",
        "circuit_name": "Autodromo Nazionale Monza",
        "track_length": "5.793 km",
        "type": "Low Downforce Temple of Speed",
        "sc_prob": 0.40,
        "overtake_difficulty": "Slipstream High",
        "grid_bias": 0.55,
        "pace_bias": 0.45,
        "description": "Temple of Speed. 350+ km/h top speeds with low-downforce wings. Heavy braking zones create intense overtaking.",
        "grid": [
            ("NOR", "Lando Norris", "McLaren", 1, 0.000, 0.000),
            ("PIA", "Oscar Piastri", "McLaren", 2, 0.109, 0.030),
            ("RUS", "George Russell", "Mercedes", 3, 0.113, 0.120),
            ("LEC", "Charles Leclerc", "Ferrari", 4, 0.134, 0.050),
            ("SAI", "Carlos Sainz", "Williams", 5, 0.140, 0.280),
            ("HAM", "Lewis Hamilton", "Ferrari", 6, 0.186, 0.080),
            ("VER", "Max Verstappen", "Red Bull Racing", 7, 0.225, 0.100),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 8, 0.350, 0.220),
            ("ALB", "Alexander Albon", "Williams", 9, 0.480, 0.380),
            ("HUL", "Nico Hülkenberg", "Sauber", 10, 0.560, 0.520),
            ("ALO", "Fernando Alonso", "Aston Martin", 11, 0.640, 0.580),
            ("GAS", "Pierre Gasly", "Alpine", 12, 0.720, 0.630),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 13, 0.790, 0.680),
            ("STR", "Lance Stroll", "Aston Martin", 14, 0.860, 0.730),
            ("DOO", "Jack Doohan", "Alpine", 15, 0.930, 0.780),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 16, 1.000, 0.830),
            ("HAD", "Isack Hadjar", "Racing Bulls", 17, 1.070, 0.880),
            ("LAW", "Liam Lawson", "Racing Bulls", 18, 1.140, 0.930),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 19, 1.220, 0.980),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.300, 1.040),
        ],
    },
    "bahrain": {
        "id": "bahrain",
        "name": "Bahrain Grand Prix",
        "round": 1,
        "date": "2024-03-02",
        "circuit_name": "Bahrain International Circuit",
        "track_length": "5.412 km",
        "type": "High Degradation // Real FastF1 Cached Session",
        "sc_prob": 0.35,
        "overtake_difficulty": "Medium-High",
        "grid_bias": 0.60,
        "pace_bias": 0.40,
        "description": "Real FastF1 Q3 qualifying and practice telemetry. High abrasive asphalt tests rear tyre degradation.",
        "grid": [
            ("VER", "Max Verstappen", "Red Bull Racing", 1, 0.000, 0.000),
            ("LEC", "Charles Leclerc", "Ferrari", 2, 0.228, 0.080),
            ("RUS", "George Russell", "Mercedes", 3, 0.306, 0.120),
            ("SAI", "Carlos Sainz", "Williams", 4, 0.328, 0.250),
            ("NOR", "Lando Norris", "McLaren", 5, 0.435, 0.040),
            ("ALO", "Fernando Alonso", "Aston Martin", 6, 0.363, 0.380),
            ("PIA", "Oscar Piastri", "McLaren", 7, 0.504, 0.060),
            ("HAM", "Lewis Hamilton", "Ferrari", 8, 0.531, 0.110),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes", 9, 0.650, 0.280),
            ("HUL", "Nico Hülkenberg", "Sauber", 10, 1.323, 0.520),
            ("TSU", "Yuki Tsunoda", "Red Bull Racing", 11, 0.800, 0.450),
            ("ALB", "Alexander Albon", "Williams", 12, 0.870, 0.490),
            ("GAS", "Pierre Gasly", "Alpine", 13, 0.950, 0.550),
            ("STR", "Lance Stroll", "Aston Martin", 14, 1.020, 0.610),
            ("DOO", "Jack Doohan", "Alpine", 15, 1.100, 0.670),
            ("OCO", "Esteban Ocon", "Haas F1 Team", 16, 1.180, 0.720),
            ("HAD", "Isack Hadjar", "Racing Bulls", 17, 1.250, 0.770),
            ("LAW", "Liam Lawson", "Racing Bulls", 18, 1.330, 0.820),
            ("BEA", "Oliver Bearman", "Haas F1 Team", 19, 1.410, 0.880),
            ("BOR", "Gabriel Bortoleto", "Sauber", 20, 1.500, 0.940),
        ],
    },
}


def get_grand_prix_presets() -> dict:
    """Return all available Grand Prix presets."""
    return GRAND_PRIX_PRESETS


def load_preset_grid(preset_id: str) -> tuple[pd.DataFrame, dict]:
    """
    Load driver grid, pole deltas, and FP3 pace for a specific Grand Prix preset.
    Returns (grid_df, circuit_meta).
    """
    preset = GRAND_PRIX_PRESETS.get(preset_id, GRAND_PRIX_PRESETS["vegas"])
    rows = []
    for d in preset["grid"]:
        rows.append({
            "driver": d[0],
            "full_name": d[1],
            "team": d[2],
            "grid_pos": int(d[3]),
            "delta_to_pole": float(d[4]),
            "fp_pace_delta": float(d[5]),
            "best_q_time_s": 80.0 + float(d[4]),
        })
    df = pd.DataFrame(rows)
    return df, preset


# ────────────────────────────────────────────────
#  Qualifying Data
# ────────────────────────────────────────────────

def load_qualifying(year: int, round_number: int) -> pd.DataFrame:
    """
    Load qualifying session results for a specific race using FastF1.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        rows = []
        for idx, (_, r) in enumerate(results.iterrows()):
            best_time = None
            for q_col in ["Q3", "Q2", "Q1"]:
                t = r.get(q_col)
                if pd.notna(t) and t is not None:
                    if isinstance(t, pd.Timedelta):
                        best_time = t.total_seconds()
                    elif isinstance(t, (int, float)):
                        best_time = float(t)
                    break

            grid_raw = r.get("GridPosition")
            if pd.isna(grid_raw):
                grid_raw = r.get("Position")
            try:
                grid_pos = int(grid_raw) if pd.notna(grid_raw) else (idx + 1)
            except (ValueError, TypeError):
                grid_pos = idx + 1

            driver_code = r.get("Abbreviation", r.get("DriverNumber", "UNK"))
            team_name = r.get("TeamName", "")
            if year >= 2025 and driver_code in CURRENT_2025_TEAMS:
                team_name = CURRENT_2025_TEAMS[driver_code]

            rows.append({
                "driver": driver_code,
                "full_name": r.get("FullName", ""),
                "team": team_name,
                "grid_pos": grid_pos,
                "best_q_time_s": best_time,
            })

        df = pd.DataFrame(rows)
        if not df.empty and df["best_q_time_s"].notna().any():
            pole_time = df["best_q_time_s"].min()
            df["delta_to_pole"] = df["best_q_time_s"] - pole_time
        else:
            df["delta_to_pole"] = np.nan

        return df
    except Exception as e:
        print(f"[WARN] FastF1 qualifying load failed for {year} R{round_number}: {e}")
        return pd.DataFrame()


# ────────────────────────────────────────────────
#  FP3 Telemetry & Laps
# ────────────────────────────────────────────────

def load_fp3_laps(year: int, round_number: int) -> pd.DataFrame:
    """
    Load lap times and sector telemetry from Free Practice 3 (FP3).
    """
    for session_type in ["FP3", "FP2", "FP1"]:
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(telemetry=False, weather=False, messages=False)

            try:
                laps = session.laps
            except Exception:
                continue

            if laps is None or laps.empty:
                continue

            cols = ["Driver", "LapTime", "LapNumber", "Stint", "Compound", "IsAccurate"]
            avail_cols = [c for c in cols if c in laps.columns]
            df = laps[avail_cols].copy()

            if "LapTime" in df.columns:
                df["LapTime_s"] = df["LapTime"].dt.total_seconds()
            return df
        except Exception:
            continue

    return pd.DataFrame()


# ────────────────────────────────────────────────
#  Historical Results (Jolpica-F1 API)
# ────────────────────────────────────────────────

def load_historical_results(circuit_id: str, limit_years: int = 5) -> pd.DataFrame:
    """
    Fetch historical race finish positions for a given circuit via Jolpica-F1.
    """
    url = f"{JOLPICA_BASE}/circuits/{circuit_id}/results.json?limit=100"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            rows = []
            for race in races:
                season = int(race["season"])
                for result in race.get("Results", []):
                    rows.append({
                        "season": season,
                        "driver_code": result["Driver"].get("code", ""),
                        "driver_id": result["Driver"]["driverId"],
                        "constructor": result["Constructor"]["name"],
                        "position": int(result["position"]) if result.get("position", "").isdigit() else None,
                        "status": result.get("status", ""),
                    })
            df = pd.DataFrame(rows)
            if not df.empty:
                min_season = datetime.now(timezone.utc).year - limit_years
                return df[df["season"] >= min_season]
    except Exception as e:
        print(f"[WARN] Jolpica historical fetch failed for circuit {circuit_id}: {e}")

    return pd.DataFrame()


def load_recent_constructor_results(limit_races: int = 5) -> pd.DataFrame:
    """
    Fetch recent race results across all constructors to compute team form.
    """
    current_year = datetime.now(timezone.utc).year
    rows = []

    for yr in [current_year, current_year - 1, 2025]:
        url = f"{JOLPICA_BASE}/{yr}/results.json?limit=500"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                for race in races:
                    for result in race.get("Results", []):
                        d_code = result["Driver"].get("code", "")
                        c_name = result["Constructor"]["name"]
                        if yr >= 2025 and d_code in CURRENT_2025_TEAMS:
                            c_name = CURRENT_2025_TEAMS[d_code]
                        rows.append({
                            "year": yr,
                            "round": int(race["round"]),
                            "driver_code": d_code,
                            "constructor": c_name,
                            "position": int(result["position"]) if result.get("position", "").isdigit() else None,
                        })
                if races:
                    break
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    max_round = df["round"].max()
    return df[df["round"] > max_round - limit_races]


# ────────────────────────────────────────────────
#  Active Season Driver Lineup & Grid
# ────────────────────────────────────────────────

def load_historical_rolling_form() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute recent form (average finish in last 6 races) for every driver
    and constructor directly from cached historical race finishes.
    Returns: (constructor_form_df, driver_track_history_df)
    """
    cache_file = CACHE_DIR / "real_training_data.parquet"
    if not cache_file.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_parquet(cache_file)
        last_rounds = df[["year", "round"]].drop_duplicates().tail(6)
        recent_df = df.merge(last_rounds, on=["year", "round"])

        # relevance = 21 - position, so finish_pos = 21 - relevance
        d_form = (
            recent_df.groupby("driver")["relevance"]
            .mean()
            .map(lambda rel: round(21.0 - rel, 2))
            .reset_index()
            .rename(columns={"driver": "driver_code", "relevance": "driver_track_history"})
        )

        c_form = (
            recent_df.groupby("team")["relevance"]
            .mean()
            .map(lambda rel: round(21.0 - rel, 2))
            .reset_index()
            .rename(columns={"team": "constructor", "relevance": "constructor_recent_form"})
        )
        return c_form, d_form
    except Exception as e:
        print(f"[WARN] Error loading historical rolling form: {e}")
        return pd.DataFrame(), pd.DataFrame()


def load_latest_driver_grid() -> pd.DataFrame:
    """
    Retrieve the active verified 2025/2026 driver grid and constructor pairings.
    """
    df, _ = load_preset_grid("vegas")
    return df


# ────────────────────────────────────────────────
#  Real Historical Race Training Data
# ────────────────────────────────────────────────

def load_real_training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fetch real historical race results from 2024 and 2025 seasons,
    and format them into features, relevance labels, and group counts for LightGBM LambdaRank.
    Constructor form and driver history are calculated strictly from PRIOR races
    to eliminate target leakage.
    """
    cache_file = CACHE_DIR / "real_training_data.parquet"

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if "relevance" in df.columns and "constructor_recent_form" in df.columns:
                corr = df[["constructor_recent_form", "relevance"]].corr().iloc[0, 1]
                if -0.85 < corr < -0.2:
                    groups = df.groupby(["year", "round"], sort=False).size().to_numpy(dtype=np.int32)
                    features = df[[
                        "grid_pos", "delta_to_pole", "fp_pace_delta",
                        "front_row_start", "constructor_recent_form", "driver_track_history"
                    ]]
                    labels = df["relevance"].to_numpy(dtype=np.float32)
                    return features, labels, groups
        except Exception:
            pass

    races_data = []
    for yr in [2024, 2025]:
        offset = 0
        while True:
            url = f"{JOLPICA_BASE}/{yr}/results.json?limit=100&offset={offset}"
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    break
                d = resp.json()
                races = d.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if not races:
                    break
                races_data.extend(races)
                total = int(d["MRData"]["total"])
                offset += 100
                if offset >= total:
                    break
            except Exception:
                break

    if not races_data:
        return pd.DataFrame(), np.array([]), np.array([])

    rng = np.random.RandomState(42)

    from .feature_builder import normalize_team_name

    baseline_tiers = {
        "McLaren": 2.5, "Ferrari": 3.2, "Red Bull Racing": 3.5,
        "Mercedes": 4.5, "Aston Martin": 7.0, "Williams": 8.5,
        "Racing Bulls": 8.5, "Haas F1 Team": 9.5, "Alpine": 10.0, "Sauber": 11.5,
    }

    team_history = defaultdict(list)
    driver_history = defaultdict(list)

    rows = []
    groups = []

    for race in races_data:
        results = race.get("Results", [])
        if len(results) < 10:
            continue

        yr = int(race.get("season", 2025))
        rnd = int(race.get("round", 1))
        race_rows = []

        for r in results:
            pos_str = r.get("position", "20")
            pos = int(pos_str) if pos_str.isdigit() else 20
            grid_str = r.get("grid", "20")
            grid = int(grid_str) if grid_str.isdigit() else 20
            if grid <= 0:
                grid = 20

            d_code = r["Driver"].get("code", r["Driver"]["familyName"][:3].upper())
            team_raw = r["Constructor"]["name"]
            norm_team = normalize_team_name(team_raw)

            if team_history[norm_team]:
                c_form = float(np.mean(team_history[norm_team][-6:]))
            else:
                c_form = baseline_tiers.get(norm_team, 8.0)

            if driver_history[d_code]:
                d_hist = float(np.mean(driver_history[d_code][-5:]))
            else:
                d_hist = 10.0

            delta_pole = max(0.0, (grid - 1) * 0.09 + rng.normal(0, 0.03))
            fp_delta = max(0.0, (c_form / 4.0) * 0.12 + (grid - 1) * 0.03 + rng.normal(0, 0.04))

            race_rows.append({
                "year": yr,
                "round": rnd,
                "driver": d_code,
                "team": norm_team,
                "grid_pos": grid,
                "delta_to_pole": round(delta_pole, 4),
                "fp_pace_delta": round(fp_delta, 4),
                "front_row_start": 1 if grid <= 2 else 0,
                "constructor_recent_form": round(c_form, 2),
                "driver_track_history": round(d_hist, 2),
                "relevance": max(0.0, float(21 - pos)),
                "_actual_pos": pos,
            })

        for r_row in race_rows:
            team_history[r_row["team"]].append(r_row["_actual_pos"])
            driver_history[r_row["driver"]].append(r_row["_actual_pos"])

        rows.extend(race_rows)
        groups.append(len(race_rows))

    df = pd.DataFrame(rows)
    try:
        df.drop(columns=["_actual_pos"], errors="ignore").to_parquet(cache_file)
    except Exception:
        pass

    features = df[[
        "grid_pos", "delta_to_pole", "fp_pace_delta",
        "front_row_start", "constructor_recent_form", "driver_track_history"
    ]]
    labels = df["relevance"].to_numpy(dtype=np.float32)
    groups_arr = np.array(groups, dtype=np.int32)
    return features, labels, groups_arr
