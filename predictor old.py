import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# List of companies
companies = [
    "Reliance",
    "TCS",
    "Infosys",
    "HDFC",
    "ICICI",
    "SBI",
    "Wipro",
    "HCL",
    "ITC",
    "AxisBank",
    "Maruti",
    "BajajFinance",
    "LT"
]


def recommend_companies(investment_amount, investment_period, expected_return):

    results = []

    for company in companies:

        model_path = f"models/{company}.keras"
        scaler_path = f"scalers/{company}_scaler.pkl"
        data_path = f"dataset/{company}_Features.csv"

        # Skip if required files are missing
        if not (
            os.path.exists(model_path)
            and os.path.exists(scaler_path)
            and os.path.exists(data_path)
        ):
            print(f"Skipping {company} (Missing files)")
            continue

        try:
            # Load model and scaler
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            # Load dataset
            df = pd.read_csv(data_path)

            if len(df) < 60:
                print(f"Skipping {company} (Not enough data)")
                continue

            # Last 60 closing prices
            last_60 = df["Close"].values[-60:]
            last_60 = last_60.reshape(-1, 1)

            # Scale data
            last_60_scaled = scaler.transform(last_60)

            # Prepare input
            X_test = np.array([last_60_scaled])

            # Predict next price
            prediction = model.predict(X_test, verbose=0)

            # Convert back to original price
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            current_price = df["Close"].iloc[-1]

            # Predicted return (%)
            predicted_return = (
                (predicted_price - current_price)
                / current_price
            ) * 100

            # Estimated investment value
            estimated_value = investment_amount * (
                1 + predicted_return / 100
            )

            # Recommendation
            if predicted_return >= expected_return:
                recommendation = "🟢 BUY"
            elif predicted_return >= expected_return * 0.5:
                recommendation = "🟡 HOLD"
            else:
                recommendation = "🔴 AVOID"

            # Store results
            results.append({
                "Company": company,
                "Current Price": round(current_price, 2),
                "Predicted Price": round(predicted_price, 2),
                "Predicted Return (%)": round(predicted_return, 2),
                "Estimated Value (₹)": round(estimated_value, 2),
                "Recommendation": recommendation
            })

        except Exception as e:
            print(f"Skipping {company}: {e}")
            continue

    # Create DataFrame
    recommendation = pd.DataFrame(results)

    if recommendation.empty:
        return recommendation

    # Sort by highest predicted return
    recommendation = recommendation.sort_values(
        by="Predicted Return (%)",
        ascending=False
    )

    recommendation.reset_index(drop=True, inplace=True)

    return recommendation