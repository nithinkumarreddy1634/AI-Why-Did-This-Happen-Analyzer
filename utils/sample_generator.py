"""Sample dataset generator for testing and user demonstration.

Generates realistic benchmark datasets for churn, sales, quality stress testing,
and multi-format testing (CSV, Excel).
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_sample_dataset(
    dataset_type: str,
    output_path: Optional[Path] = None,
    n_rows: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate benchmark datasets for demonstration and testing."""
    np.random.seed(random_seed)

    if dataset_type == "churn":
        customer_ids = [f"CUST-{1000 + i}" for i in range(n_rows)]
        ages = np.random.randint(18, 75, size=n_rows)
        tenure = np.random.randint(1, 72, size=n_rows)
        monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_rows), 2)
        total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, size=n_rows), 2)
        total_charges = np.clip(total_charges, 20.0, None)
        contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
        tech_support = np.random.choice(["Yes", "No", "No internet"], size=n_rows, p=[0.35, 0.50, 0.15])
        complaints = np.random.poisson(lam=0.8, size=n_rows)
        
        # Churn probability higher with high charges, complaints, and month-to-month
        churn_prob = 0.15 + (complaints * 0.12) + (monthly_charges / 300) - (tenure / 150)
        churn_prob = np.clip(churn_prob, 0.05, 0.90)
        churn = ["Yes" if np.random.rand() < p else "No" for p in churn_prob]

        df = pd.DataFrame({
            "CustomerID": customer_ids,
            "Age": ages,
            "TenureMonths": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contracts,
            "TechSupport": tech_support,
            "ComplaintsCount": complaints,
            "Churn": churn,
        })

    elif dataset_type == "sales":
        deal_ids = [f"DEAL-{5000 + i}" for i in range(n_rows)]
        regions = np.random.choice(["North America", "EMEA", "APAC", "LATAM"], size=n_rows, p=[0.4, 0.3, 0.2, 0.1])
        rep_exp = np.random.randint(1, 15, size=n_rows)
        discounts = np.round(np.random.uniform(0.0, 0.35, size=n_rows), 2)
        lead_sources = np.random.choice(["Inbound Web", "Outbound Cold", "Partner Referral", "Event"], size=n_rows)
        marketing_spend = np.round(np.random.uniform(500.0, 5000.0, size=n_rows), 2)
        units_sold = np.random.randint(5, 100, size=n_rows)
        
        # Revenue depends on units sold, rep experience, and discount
        revenue = np.round(
            (units_sold * 150.0) * (1.0 - discounts) + (rep_exp * 250.0) + (marketing_spend * 0.8) + np.random.normal(0, 200, size=n_rows),
            2,
        )

        df = pd.DataFrame({
            "DealID": deal_ids,
            "Region": regions,
            "RepExperienceYears": rep_exp,
            "DiscountPercent": discounts,
            "LeadSource": lead_sources,
            "MarketingSpend": marketing_spend,
            "UnitsSold": units_sold,
            "MonthlyRevenue": revenue,
        })

    elif dataset_type == "stress_test":
        records = 500
        emp_ids = [f"EMP-{i:04d}" for i in range(records)]
        departments = np.random.choice(["Sales", "Engineering", "Marketing", "HR", "Support"], size=records)
        salaries = np.round(np.random.uniform(45000, 150000, size=records), 2)
        ratings = np.random.choice([1, 2, 3, 4, 5, np.nan], size=records, p=[0.1, 0.15, 0.4, 0.2, 0.1, 0.05])
        signup_dates = pd.date_range(start="2022-01-01", periods=records, freq="D").strftime("%Y-%m-%d")
        country = ["United States"] * records  # Constant column
        notes = ["High performer" if r == 5 else "Standard evaluation note" for r in ratings]

        df = pd.DataFrame({
            "EmployeeID": emp_ids,
            "Department": departments,
            "Salary": salaries,
            "PerformanceRating": ratings,
            "HireDate": signup_dates,
            "Country": country,
            "ReviewNotes": notes,
        })

        # Inject some duplicates
        dup_indices = [5, 12, 25, 40, 55]
        dups = df.iloc[dup_indices].copy()
        df = pd.concat([df, dups], ignore_index=True)

        # Inject some missing values
        df.loc[10:20, "Salary"] = np.nan
        df.loc[30:35, "Department"] = np.nan

    else:
        # Default simple generic tabular dataset
        df = pd.DataFrame({
            "Feature_A": np.random.randn(n_rows),
            "Feature_B": np.random.randint(0, 100, size=n_rows),
            "Target": np.random.choice(["Class_0", "Class_1"], size=n_rows),
        })

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df.to_excel(p, index=False, engine="openpyxl")
        else:
            df.to_csv(p, index=False)

    return df
