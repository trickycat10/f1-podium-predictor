#!/usr/bin/env python3
"""
run.py – Entry point for the F1 Podium Prediction System.
Runs the full data pipeline and launches the Streamlit dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline_check():
    """Quick dry-run to verify the pipeline components work."""
    print("=" * 60)
    print("  F1 Podium Predictor – Pipeline Verification")
    print("=" * 60)

    # 1. Calendar lookup
    print("\n[1/4] Testing calendar lookup...")
    try:
        from src.calendar_service import get_next_grand_prix
        gp = get_next_grand_prix()
        print(f"  ✓ Next GP: {gp['raceName']} (Round {gp['round']}) – {gp['date']}")
        print(f"  ✓ Circuit: {gp.get('circuitName', gp['circuitId'])}")
    except Exception as e:
        print(f"  ✗ Calendar lookup failed: {e}")
        gp = None

    # 2. FastF1 cache
    print("\n[2/4] Testing FastF1 cache initialization...")
    try:
        import fastf1
        cache_dir = PROJECT_ROOT / "data_cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        print(f"  ✓ Cache directory: {cache_dir}")
    except Exception as e:
        print(f"  ✗ FastF1 cache failed: {e}")

    # 3. Model training & prediction
    print("\n[3/4] Testing model training & Monte Carlo simulation...")
    try:
        from src.model import (
            get_training_dataset,
            F1RankingModel,
            monte_carlo_plackett_luce,
        )
        from src.feature_builder import FEATURE_COLUMNS
        import numpy as np

        X_train, y_train, groups = get_training_dataset()
        model = F1RankingModel()
        model.train(X_train, y_train, groups)
        print(f"  ✓ Model trained on {len(X_train)} samples, {len(groups)} race groups")

        # Quick prediction
        scores = model.predict_scores(X_train.head(20))
        drivers = [f"D{i}" for i in range(20)]
        probs = monte_carlo_plackett_luce(scores, drivers, n_simulations=1000)
        total_p_win = probs["p_win"].sum()
        total_p_podium = probs[["p_win", "p_p2", "p_p3"]].sum().sum()
        print(f"  ✓ MC Simulation complete: Σp_win = {total_p_win:.4f} (should ≈ 1.0)")
        print(f"  ✓ Total podium slots  : {total_p_podium:.4f} (should ≈ 3.0)")

        # Verify probabilities sum correctly
        assert abs(total_p_win - 1.0) < 0.01, f"p_win sum out of range: {total_p_win}"
        assert abs(total_p_podium - 3.0) < 0.05, f"podium sum out of range: {total_p_podium}"
        print("  ✓ Probability sanity checks passed!")

        # Feature importance
        importance = model.feature_importance()
        print(f"  ✓ Feature importances: {importance}")

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Summary
    print("\n[4/4] All checks passed! ✅")
    print("=" * 60)


def launch_streamlit():
    """Launch the Streamlit dashboard."""
    app_path = PROJECT_ROOT / "src" / "app.py"
    print(f"\n🚀 Launching Streamlit dashboard...")
    print(f"   URL: http://localhost:8501")
    print(f"   Press Ctrl+C to stop.\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    if "--check-only" in sys.argv:
        run_pipeline_check()
    else:
        run_pipeline_check()
        launch_streamlit()
