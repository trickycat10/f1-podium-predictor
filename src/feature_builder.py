"""
feature_builder.py – Compute FP3 race-pace deltas, qualifying deltas,
and engineer all features needed for the LTR ranking model.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_fp3_pace_delta(fp3_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out in/out laps, isolate long-run stints (>= 3 consecutive
    timed laps), and compute each driver's median lap pace relative
    to the session leader.

    Returns a DataFrame with columns: driver, team, fp_pace_delta, long_run_median.
    """
    if fp3_laps.empty:
        return pd.DataFrame(columns=["driver", "team", "fp_pace_delta", "long_run_median"])

    df = fp3_laps.copy()
    df = df.dropna(subset=["lap_time_s"])

    # Filter out likely in/out laps (> 1.2× session median) and very short laps
    session_median = df["lap_time_s"].median()
    if pd.isna(session_median) or session_median <= 0:
        return pd.DataFrame(columns=["driver", "team", "fp_pace_delta", "long_run_median"])

    df = df[
        (df["lap_time_s"] < session_median * 1.15) &
        (df["lap_time_s"] > session_median * 0.90)
    ]

    # Identify long-run stints: >= 3 consecutive timed laps in the same stint
    long_run_rows = []
    for (driver, stint), grp in df.groupby(["driver", "stint"]):
        grp = grp.sort_values("lap_number")
        if len(grp) >= 3:
            long_run_rows.append(grp)

    if not long_run_rows:
        # Fallback: use all valid laps
        long_run_df = df
    else:
        long_run_df = pd.concat(long_run_rows, ignore_index=True)

    # Compute driver-level median pace
    driver_pace = (
        long_run_df.groupby(["driver", "team"])["lap_time_s"]
        .median()
        .reset_index()
        .rename(columns={"lap_time_s": "long_run_median"})
    )

    # Delta to session leader
    leader_pace = driver_pace["long_run_median"].min()
    driver_pace["fp_pace_delta"] = driver_pace["long_run_median"] - leader_pace

    return driver_pace[["driver", "team", "fp_pace_delta", "long_run_median"]]


def normalize_team_name(name: str) -> str:
    """Standardize constructor names across different F1 data sources."""
    if not isinstance(name, str):
        return ""
    n = name.lower()
    if "mclaren" in n:
        return "McLaren"
    if "ferrari" in n:
        return "Ferrari"
    if "mercedes" in n:
        return "Mercedes"
    if "red bull" in n:
        return "Red Bull Racing"
    if "williams" in n:
        return "Williams"
    if "aston" in n:
        return "Aston Martin"
    if "haas" in n:
        return "Haas F1 Team"
    if "sauber" in n or "audi" in n or "kick" in n:
        return "Sauber"
    if "alpine" in n or "renault" in n:
        return "Alpine"
    if "racing bulls" in n or "rb" in n or "toro rosso" in n or "alphatauri" in n:
        return "Racing Bulls"
    return name.strip()


def compute_constructor_form(recent_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple constructor recent form score:
    average finishing position across recent races (lower = better).
    """
    if recent_results.empty:
        return pd.DataFrame(columns=["constructor", "constructor_recent_form"])

    rr = recent_results.copy()
    if "constructor" in rr.columns:
        rr["constructor"] = rr["constructor"].apply(normalize_team_name)

    team_form = (
        rr.dropna(subset=["position"])
        .groupby("constructor")["position"]
        .mean()
        .reset_index()
        .rename(columns={"position": "constructor_recent_form"})
    )
    return team_form


def compute_driver_track_history(
    historical: pd.DataFrame, driver_mapping: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute driver-level track history score:
    average historical finishing position at this circuit (lower = better).
    """
    if historical.empty:
        return pd.DataFrame(columns=["driver_code", "driver_track_history"])

    hist = historical.dropna(subset=["position"]).copy()
    track_hist = (
        hist.groupby("driver_code")["position"]
        .mean()
        .reset_index()
        .rename(columns={"position": "driver_track_history"})
    )
    return track_hist


def build_feature_matrix(
    qualifying: pd.DataFrame,
    fp3_pace: Optional[pd.DataFrame] = None,
    constructor_form: Optional[pd.DataFrame] = None,
    track_history: Optional[pd.DataFrame] = None,
    circuit_meta: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Merge all data sources into a single feature matrix per driver.

    Features produced:
    - grid_pos: qualifying grid position
    - delta_to_pole: qualifying time delta to pole (seconds)
    - fp_pace_delta: FP3 long-run pace delta to leader
    - front_row_start: 1 if grid_pos <= 2, else 0
    - constructor_recent_form: avg recent finish position of team
    - driver_track_history: avg historical finish at this track
    """
    if fp3_pace is None:
        fp3_pace = pd.DataFrame()
    if constructor_form is None or constructor_form.empty or track_history is None or track_history.empty:
        try:
            from .data_loader import load_historical_rolling_form
            c_auto, d_auto = load_historical_rolling_form()
            if (constructor_form is None or constructor_form.empty) and not c_auto.empty:
                constructor_form = c_auto
            if (track_history is None or track_history.empty) and not d_auto.empty:
                track_history = d_auto
        except Exception:
            pass

    if constructor_form is None:
        constructor_form = pd.DataFrame()
    if track_history is None:
        track_history = pd.DataFrame()

    df = qualifying.copy()
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalize_team_name)

    # Merge FP3 pace
    if not fp3_pace.empty and "fp_pace_delta" in fp3_pace.columns:
        df = df.merge(fp3_pace[["driver", "fp_pace_delta"]], on="driver", how="left")
    else:
        # Realistic proxy based on qualifying delta when practice session hasn't occurred yet
        df["fp_pace_delta"] = df["delta_to_pole"].fillna(0.5) * 0.75 + 0.05

    # Front row indicator
    df["front_row_start"] = (df["grid_pos"] <= 2).astype(int)

    # Merge constructor form
    if not constructor_form.empty:
        df = df.merge(
            constructor_form, left_on="team", right_on="constructor", how="left"
        )
        df.drop(columns=["constructor"], errors="ignore", inplace=True)
    else:
        df["constructor_recent_form"] = np.nan

    # Merge track history
    if not track_history.empty:
        df = df.merge(
            track_history, left_on="driver", right_on="driver_code", how="left"
        )
        df.drop(columns=["driver_code"], errors="ignore", inplace=True)
    else:
        df["driver_track_history"] = np.nan

    # Realistic modern baselines for constructors and drivers
    team_baseline = {
        "McLaren": 2.5, "Ferrari": 3.2, "Red Bull Racing": 3.5,
        "Mercedes": 4.5, "Aston Martin": 7.0, "Williams": 8.5,
        "Racing Bulls": 8.5, "Haas F1 Team": 9.5, "Alpine": 10.0, "Sauber": 11.5,
    }
    driver_baseline = {
        "VER": 1.7, "NOR": 2.5, "LEC": 3.0, "HAM": 3.2,
        "RUS": 4.5, "PIA": 6.5, "ANT": 6.8, "SAI": 7.0, "ALO": 7.5, "ALB": 8.0,
    }

    df["constructor_recent_form"] = df["constructor_recent_form"].fillna(df["team"].map(team_baseline).fillna(8.0))
    df["driver_track_history"] = df["driver_track_history"].fillna(df["driver"].map(driver_baseline).fillna(9.0))

    # Fill NaNs with realistic defaults
    df["fp_pace_delta"] = df["fp_pace_delta"].fillna(df["delta_to_pole"] * 0.75 + 0.05)
    df["delta_to_pole"] = df["delta_to_pole"].fillna(df["grid_pos"].map(lambda g: round((g - 1) * 0.09, 3)))

    # Apply circuit physics modulation if circuit_meta provided
    if circuit_meta and "grid_bias" in circuit_meta:
        grid_bias = circuit_meta.get("grid_bias", 0.60)
        # On circuits with extreme qualifying importance (e.g. Monaco), grid position and pole delta are heightened
        df["delta_to_pole"] = df["delta_to_pole"] * (grid_bias / 0.60)

    return df


FEATURE_COLUMNS = [
    "grid_pos",
    "delta_to_pole",
    "fp_pace_delta",
    "front_row_start",
    "constructor_recent_form",
    "driver_track_history",
]
