"""
model.py – LightGBM Learning-to-Rank model with Monte Carlo Plackett-Luce
simulation for podium probability estimation.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from typing import Optional
import warnings

from .feature_builder import FEATURE_COLUMNS, normalize_team_name

warnings.filterwarnings("ignore", category=UserWarning)


# Modern 2025/2026 Team Hierarchy & Realistic Constructor Baselines
MODERN_TEAM_TIERS = {
    "McLaren": 2.5,
    "Ferrari": 3.2,
    "Red Bull Racing": 3.5,
    "Mercedes": 4.5,
    "Aston Martin": 7.0,
    "Williams": 8.5,
    "Racing Bulls": 8.5,
    "Haas F1 Team": 9.5,
    "Alpine": 10.0,
    "Sauber": 11.5,
}


# ────────────────────────────────────────────────
#  Synthetic Historical Seed Data (2024–2026)
# ────────────────────────────────────────────────

def generate_seed_training_data(n_races: int = 60) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate synthetic but realistic training data for the LTR model
    so it works out-of-the-box even without a populated cache.
    Reflects modern F1 dynamics: pole advantage, front row conversion,
    and constructor performance tiers.
    """
    rng = np.random.RandomState(42)
    rows = []
    groups = []

    teams = list(MODERN_TEAM_TIERS.keys())
    drivers_per_team = 2

    for race_id in range(n_races):
        n_drivers = len(teams) * drivers_per_team
        grid = rng.permutation(n_drivers) + 1

        for i, team in enumerate(teams):
            for seat in range(drivers_per_team):
                idx = i * drivers_per_team + seat
                g = int(grid[idx])
                tier = MODERN_TEAM_TIERS[team]

                delta_pole = max(0.0, (g - 1) * 0.08 + rng.normal(0, 0.03))
                fp_delta = max(0.0, (tier / 4.0) * 0.12 + (g - 1) * 0.03 + rng.normal(0, 0.04))
                front_row = 1 if g <= 2 else 0
                constr_form = float(tier) + rng.normal(0, 0.7)
                track_hist = float(tier) + rng.normal(0, 1.0)

                # Lower finish_score is better (P1)
                finish_score = g * 0.60 + tier * 0.35 + delta_pole * 1.5 + rng.normal(0, 1.2)
                rows.append({
                    "race_id": race_id,
                    "grid_pos": g,
                    "delta_to_pole": round(delta_pole, 4),
                    "fp_pace_delta": round(fp_delta, 4),
                    "front_row_start": front_row,
                    "constructor_recent_form": round(max(1.0, constr_form), 2),
                    "driver_track_history": round(max(1.0, track_hist), 2),
                    "_finish_score": finish_score,
                })

        groups.append(n_drivers)

    df = pd.DataFrame(rows)
    df["relevance"] = 0.0

    # Assign relevance per race correctly: P1 gets highest relevance (20.0), P20 gets 1.0
    for race_id in range(n_races):
        mask = df["race_id"] == race_id
        race_df = df[mask]
        ranks = race_df["_finish_score"].rank(ascending=True)
        df.loc[mask, "relevance"] = (len(race_df) + 1 - ranks).values

    labels_arr = df["relevance"].to_numpy(dtype=np.float32)
    groups_arr = np.array(groups, dtype=np.int32)
    feature_df = df[FEATURE_COLUMNS].copy()

    return feature_df, labels_arr, groups_arr


# ────────────────────────────────────────────────
#  LTR Model Training
# ────────────────────────────────────────────────

class F1RankingModel:
    """LightGBM LambdaRank model for F1 driver ranking."""

    def __init__(self):
        self.model: Optional[lgb.LGBMRanker] = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        n_estimators: int = 150,
        learning_rate: float = 0.04,
        num_leaves: int = 15,
    ):
        """
        Train the LambdaRank model with domain-specific monotonic constraints
        to ensure physical consistency (e.g. pole position advantage, race pace,
        and constructor form strictly reward ranking scores).
        """
        # Monotone constraints for FEATURE_COLUMNS:
        # grid_pos (-1), delta_to_pole (-1), fp_pace_delta (-1),
        # front_row_start (+1), constructor_recent_form (-1), driver_track_history (-1)
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=12,
            monotone_constraints=[-1, -1, -1, 1, -1, -1],
            reg_alpha=0.1,
            reg_lambda=1.0,
            importance_type="gain",
            verbose=-1,
            force_col_wise=True,
        )

        train_feats = X[FEATURE_COLUMNS].copy() if all(c in X.columns for c in FEATURE_COLUMNS) else X
        self.model.fit(
            train_feats, y,
            group=groups,
        )
        self.is_trained = True
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores for a set of drivers."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained yet.")
        if all(col in X.columns for col in FEATURE_COLUMNS):
            features = X[FEATURE_COLUMNS].copy()
        else:
            features = X
        return self.model.predict(features)

    def feature_importance(self) -> dict:
        """Return feature importances."""
        if not self.is_trained or self.model is None:
            return {}
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))


# ────────────────────────────────────────────────
#  Monte Carlo Plackett-Luce Simulation
# ────────────────────────────────────────────────

def monte_carlo_plackett_luce(
    scores: np.ndarray,
    driver_names: list[str],
    n_simulations: int = 10_000,
    gumbel_scale: float = 0.55,
    dnf_base_rate: float = 0.04,
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run a Plackett-Luce simulation with Gumbel noise to estimate podium
    probabilities, DNF rates, and expected finish positions for each driver.

    The scores are standardized so that differences calibrate consistently
    with the Gumbel distribution, modeling race-day uncertainty: safety cars,
    strategy variance, mechanical DNFs, weather, and tyre degradation.

    Parameters
    ----------
    scores : array of LTR model scores (higher = predicted better finish)
    driver_names : list of driver abbreviations
    n_simulations : number of Monte Carlo samples
    gumbel_scale : noise scale (higher = more chaos/variance)
    dnf_base_rate : baseline probability of driver retirement
    rng_seed : random seed (None for live dynamic Monte Carlo sampling)

    Returns
    -------
    DataFrame with columns: driver, p_win, p_p2, p_p3, p_podium, p_dnf, exp_finish
    """
    rng = np.random.RandomState(rng_seed)
    n_drivers = len(scores)

    # Standardize scores so relative gaps calibrate cleanly against Gumbel noise scale
    score_std = np.std(scores)
    if score_std > 1e-6:
        norm_scores = (scores - np.mean(scores)) / score_std
    else:
        norm_scores = scores - np.mean(scores)

    p1_count = np.zeros(n_drivers, dtype=np.int64)
    p2_count = np.zeros(n_drivers, dtype=np.int64)
    p3_count = np.zeros(n_drivers, dtype=np.int64)
    dnf_count = np.zeros(n_drivers, dtype=np.int64)
    position_sum = np.zeros(n_drivers, dtype=np.float64)

    effective_dnf = max(0.01, min(0.18, dnf_base_rate * (1.0 + (gumbel_scale - 0.55))))

    for _ in range(n_simulations):
        # Simulate mechanical and collision DNFs
        dnfs = rng.rand(n_drivers) < effective_dnf
        noisy_scores = norm_scores + rng.gumbel(loc=0, scale=gumbel_scale, size=n_drivers)
        # Retired drivers cannot place
        noisy_scores[dnfs] = -9999.0
        ranking = np.argsort(-noisy_scores)

        if not dnfs[ranking[0]]:
            p1_count[ranking[0]] += 1
        if n_drivers > 1 and not dnfs[ranking[1]]:
            p2_count[ranking[1]] += 1
        if n_drivers > 2 and not dnfs[ranking[2]]:
            p3_count[ranking[2]] += 1

        dnf_count[dnfs] += 1
        for pos_idx, driver_idx in enumerate(ranking, start=1):
            position_sum[driver_idx] += pos_idx

    # Convert to probabilities & expected finish
    results = pd.DataFrame({
        "driver": driver_names,
        "p_win": p1_count / n_simulations,
        "p_p2": p2_count / n_simulations,
        "p_p3": p3_count / n_simulations,
        "p_dnf": dnf_count / n_simulations,
        "exp_finish": np.round(position_sum / n_simulations, 1),
    })
    results["p_podium"] = results["p_win"] + results["p_p2"] + results["p_p3"]

    # Sort by total podium probability descending
    results = results.sort_values("p_podium", ascending=False).reset_index(drop=True)

    return results


def simulate_single_grand_prix(
    scores: np.ndarray,
    drivers_df: pd.DataFrame,
    gumbel_scale: float = 0.55,
    dnf_rate: float = 0.05,
    rng_seed: Optional[int] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Simulate one concrete, stochastic Grand Prix race realization.
    Returns:
      - classification DataFrame (finish_pos, driver, full_name, team, grid_pos, delta_grid, gap_to_leader, status)
      - race commentary bullets (key overtakes, incidents, safety car)
    """
    rng = np.random.RandomState(rng_seed)
    n_drivers = len(scores)

    score_std = np.std(scores)
    norm_scores = (scores - np.mean(scores)) / (score_std if score_std > 1e-6 else 1.0)

    # 1. Determine DNFs
    dnf_reasons = [
        "Mechanical DNF (Power Unit)",
        "Mechanical DNF (Hydraulics)",
        "Mechanical DNF (Brakes)",
        "Collision (Turn 1 Incident)",
        "Collision (Overtake Clash)",
        "Suspension Failure (Kerb Strike)",
    ]
    is_dnf = rng.rand(n_drivers) < dnf_rate
    dnf_notes = [rng.choice(dnf_reasons) if d else "Finished" for d in is_dnf]

    # 2. Sample noisy race pace with Gumbel perturbation
    noise = rng.gumbel(loc=0, scale=gumbel_scale, size=n_drivers)
    race_performances = norm_scores + noise

    # Put DNF drivers to the back
    race_performances[is_dnf] = -9999.0 + rng.uniform(-10, 0, size=is_dnf.sum())

    finish_order = np.argsort(-race_performances)

    classified_rows = []
    gap_cumulative = 0.0
    active_finish_num = 1

    for idx_in_order, driver_idx in enumerate(finish_order):
        d_row = drivers_df.iloc[driver_idx].to_dict()
        d_name = d_row.get("driver", f"D{driver_idx}")
        start_grid = int(d_row.get("grid_pos", idx_in_order + 1))
        dnf = is_dnf[driver_idx]
        status = dnf_notes[driver_idx]

        if dnf:
            finish_display = "DNF"
            pos_num = 20
            gap_str = "RETIRED"
            delta_grid = 0
        else:
            finish_display = f"P{active_finish_num}"
            pos_num = active_finish_num
            if active_finish_num == 1:
                gap_str = "WINNER"
            else:
                gap_cumulative += float(rng.uniform(1.1, 4.2))
                gap_str = f"+{gap_cumulative:.3f}s"
            delta_grid = start_grid - active_finish_num
            active_finish_num += 1

        classified_rows.append({
            "finish_pos": finish_display,
            "finish_num": pos_num,
            "driver": d_name,
            "full_name": d_row.get("full_name", d_name),
            "team": d_row.get("team", ""),
            "grid_pos": start_grid,
            "delta_grid": delta_grid,
            "gap_to_leader": gap_str,
            "status": status,
            "is_dnf": dnf,
        })

    result_df = pd.DataFrame(classified_rows).sort_values("finish_num").reset_index(drop=True)

    # Generate authentic race narrative bullets
    commentary = []
    winner = result_df.iloc[0]
    commentary.append(f"🏁 **{winner['driver']} ({winner['team']})** wins the Grand Prix starting from P{winner['grid_pos']}!")

    p2 = result_df.iloc[1]
    p3 = result_df.iloc[2]
    commentary.append(f"🍾 **Podium:** P2 {p2['driver']} ({p2['gap_to_leader']}) and P3 {p3['driver']} ({p3['gap_to_leader']}).")

    gainers = [r for _, r in result_df.iterrows() if not r['is_dnf'] and r['delta_grid'] >= 2]
    if gainers:
        top_gainer = max(gainers, key=lambda x: x['delta_grid'])
        commentary.append(f"🚀 **Driver of the Day charge:** {top_gainer['driver']} advanced {top_gainer['delta_grid']} places from P{top_gainer['grid_pos']} to {top_gainer['finish_pos']}!")

    dnfs = [r for _, r in result_df.iterrows() if r['is_dnf']]
    if dnfs:
        dnf_summary = ", ".join([f"{d['driver']} ({d['status'].split('(')[-1].rstrip(')')})" for d in dnfs])
        commentary.append(f"⚠️ **Race Incidents:** {len(dnfs)} retirement(s) — {dnf_summary}.")
    else:
        commentary.append("🛡️ **Clean Race:** All 20 drivers reached the chequered flag.")

    return result_df, commentary


def get_training_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fetch real 2024-2025 Grand Prix race training data from Jolpica API,
    falling back to modern synthetic seed data if unavailable.
    """
    try:
        from .data_loader import load_real_training_data
        X_real, y_real, groups_real = load_real_training_data()
        if not X_real.empty and len(groups_real) >= 5:
            print(f"[INFO] Training LightGBM LambdaRank on {len(X_real)} real race results ({len(groups_real)} races).")
            return X_real, y_real, groups_real
    except Exception as e:
        print(f"[WARN] Real training data load failed: {e}. Using modern seed data.")

    return generate_seed_training_data(n_races=60)


# ────────────────────────────────────────────────
#  Pipeline Convenience
# ────────────────────────────────────────────────

def train_and_predict(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, F1RankingModel]:
    """
    Full pipeline: train model on historical data, predict scores for current
    drivers, and run Monte Carlo simulation.
    """
    X_train, y_train, groups_train = get_training_dataset()

    model = F1RankingModel()
    model.train(X_train, y_train, groups_train)

    X_pred = feature_matrix[FEATURE_COLUMNS].copy()
    scores = model.predict_scores(X_pred)

    probs = monte_carlo_plackett_luce(
        scores=scores,
        driver_names=feature_matrix["driver"].tolist(),
        n_simulations=10_000,
        gumbel_scale=0.55,
    )

    meta = feature_matrix[["driver", "team", "full_name", "grid_pos",
                            "delta_to_pole", "fp_pace_delta"]].copy()
    predictions = probs.merge(meta, on="driver", how="left")
    predictions = predictions.sort_values("p_podium", ascending=False).reset_index(drop=True)

    return predictions, model
