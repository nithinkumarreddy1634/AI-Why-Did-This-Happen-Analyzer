"""Sample dataset generator for testing and interactive exploration of unexpected outcomes."""

import pandas as pd
import numpy as np


def generate_churn_investigation_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic enterprise dataset demonstrating an unexpected churn surge."""
    rng = np.random.default_rng(random_state)
    n_baseline = n_samples // 2
    n_recent = n_samples - n_baseline

    # 1. Baseline Cohort (H1: Jan - Jun)
    baseline_dates = pd.date_range("2026-01-01", "2026-06-30", periods=n_baseline)
    b_charges = rng.normal(loc=65.0, scale=12.0, size=n_baseline).clip(30.0, 110.0)
    b_tickets = rng.poisson(lam=1.2, size=n_baseline)
    b_tenure = rng.integers(low=6, high=60, size=n_baseline)
    b_age = rng.normal(loc=41.0, scale=11.0, size=n_baseline).clip(18, 75).astype(int)
    b_usage = rng.normal(loc=120.0, scale=25.0, size=n_baseline).clip(20.0, 220.0)
    b_satisfaction = rng.choice([3, 4, 5], size=n_baseline, p=[0.2, 0.5, 0.3])
    b_tier = rng.choice(["Basic", "Pro", "Enterprise"], size=n_baseline, p=[0.5, 0.35, 0.15])
    b_region = rng.choice(["North", "South", "East", "West"], size=n_baseline, p=[0.25, 0.25, 0.25, 0.25])
    # ~10% churn rate in baseline
    b_churn_prob = np.where(b_tickets > 3, 0.25, 0.08)
    b_churn = np.where(rng.uniform(size=n_baseline) < b_churn_prob, "Yes", "No")

    # 2. Recent Cohort (H2: Jul - Dec - Unexpected Surge)
    recent_dates = pd.date_range("2026-07-01", "2026-12-31", periods=n_recent)
    # Price hike: monthly charges shifted higher
    r_charges = rng.normal(loc=82.0, scale=16.0, size=n_recent).clip(35.0, 160.0)
    # Add a few extreme billing anomalies
    r_charges[0:8] = [285.0, 310.0, 290.0, 345.0, 320.0, 275.0, 360.0, 295.0]
    # Support ticket spike in West region
    r_tickets = rng.poisson(lam=2.8, size=n_recent)
    r_tenure = rng.integers(low=2, high=50, size=n_recent)
    r_age = rng.normal(loc=40.5, scale=11.0, size=n_recent).clip(18, 75).astype(int)
    # Usage dropped
    r_usage = rng.normal(loc=95.0, scale=30.0, size=n_recent).clip(10.0, 200.0)
    r_satisfaction = rng.choice([1, 2, 3, 4], size=n_recent, p=[0.35, 0.35, 0.2, 0.1])
    r_tier = rng.choice(["Basic", "Pro", "Enterprise"], size=n_recent, p=[0.45, 0.40, 0.15])
    r_region = rng.choice(["North", "South", "East", "West"], size=n_recent, p=[0.2, 0.2, 0.2, 0.4])
    # ~28% churn rate in recent cohort, concentrated in West and Pro tiers
    r_churn_prob = np.where(
        (r_region == "West") | (r_charges > 90),
        0.42,
        0.18
    )
    r_churn = np.where(rng.uniform(size=n_recent) < r_churn_prob, "Yes", "No")

    # Assemble complete DataFrame
    df_baseline = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(n_baseline)],
        "timestamp": baseline_dates,
        "cohort": "Baseline (H1)",
        "age": b_age,
        "region": b_region,
        "subscription_tier": b_tier,
        "monthly_charges": np.round(b_charges, 2),
        "tenure_months": b_tenure,
        "support_tickets": b_tickets,
        "monthly_usage_hours": np.round(b_usage, 1),
        "satisfaction_score": b_satisfaction,
        "churn": b_churn,
    })

    df_recent = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + n_baseline + i}" for i in range(n_recent)],
        "timestamp": recent_dates,
        "cohort": "Recent Surge (H2)",
        "age": r_age,
        "region": r_region,
        "subscription_tier": r_tier,
        "monthly_charges": np.round(r_charges, 2),
        "tenure_months": r_tenure,
        "support_tickets": r_tickets,
        "monthly_usage_hours": np.round(r_usage, 1),
        "satisfaction_score": r_satisfaction,
        "churn": r_churn,
    })

    full_df = pd.concat([df_baseline, df_recent], ignore_index=True)
    return full_df
