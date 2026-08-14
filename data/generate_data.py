"""
Dataset Generation Script for Customer Churn Prediction System.
Synthesizes a realistic historical customer dataset with non-linear churn dynamics.
"""

import os
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic dataset for customer churn prediction."""
    np.random.seed(random_state)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]

    # Demographics
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.52, 0.48])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # Account Info
    tenure = np.random.randint(1, 73, size=n_samples)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.35, 0.22, 0.22, 0.21],
    )

    # Services
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.45, 0.20])

    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for i in range(n_samples):
        if internet_service[i] == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.43, 0.57]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))

    # Base Monthly Charges based on features
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if internet_service[i] == "DSL":
            base += 30.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0

        if online_security[i] == "Yes": base += 5.0
        if online_backup[i] == "Yes": base += 6.0
        if device_protection[i] == "Yes": base += 7.0
        if tech_support[i] == "Yes": base += 8.0
        if streaming_tv[i] == "Yes": base += 10.0
        if streaming_movies[i] == "Yes": base += 10.0

        # Add slight random variation
        base += np.random.normal(0, 3)
        monthly_charges.append(round(max(base, 18.5), 2))

    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Churn probability log-odds formula based on real business drivers
    logit = (
        -0.8
        + 0.045 * (monthly_charges - 65.0)
        - 0.05 * tenure
        + 1.2 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.6 * (internet_service == "Fiber optic")
        + 0.7 * (payment_method == "Electronic check")
        - 0.6 * (np.array(tech_support) == "Yes")
        - 0.5 * (np.array(online_security) == "Yes")
        + 0.3 * (senior_citizen == 1)
        - 0.4 * (dependents == "Yes")
    )

    churn_prob = 1 / (1 + np.exp(-logit))
    churn = np.random.binomial(1, churn_prob)
    churn_str = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_str,
        }
    )

    return df


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "customer_churn.csv")

    df = generate_churn_dataset(n_samples=2500, random_state=42)
    df.to_csv(file_path, index=False)
    print(f"Generated dataset with {len(df)} records at: {file_path}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
