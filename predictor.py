import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

# =====================================================
# Project Paths
# =====================================================

MODEL_FOLDER = "models"
SCALER_FOLDER = "scalers"
DATASET_FOLDER = "dataset"

# =====================================================
# Companies
# (Only companies that have trained models)
# =====================================================

COMPANIES = [
    "AxisBank",
    "BajajFinance",
    "HCL",
    "HDFC",
    "ICICI",
    "Infosys",
    "ITC",
    "LT",
    "Maruti",
    "Reliance",
    "SBI",
    "TCS",
    "Wipro"
]

# =====================================================
# Dictionaries
# =====================================================

models = {}
scalers = {}
datasets = {}

# =====================================================
# Load Everything
# =====================================================

print("Loading Project Files...\n")

for company in COMPANIES:

    model_path = os.path.join(
        MODEL_FOLDER,
        f"{company}.keras"
    )

    scaler_path = os.path.join(
        SCALER_FOLDER,
        f"{company}_scaler.pkl"
    )

    data_path = os.path.join(
        DATASET_FOLDER,
        f"{company}_Features.csv"
    )

    # ------------------------
    # Model
    # ------------------------

    if os.path.exists(model_path):

        models[company] = load_model(model_path)

        print(f"✅ Model Loaded : {company}")

    else:

        print(f"❌ Missing Model : {company}")

    # ------------------------
    # Scaler
    # ------------------------

    if os.path.exists(scaler_path):

        scalers[company] = joblib.load(scaler_path)

    else:

        print(f"❌ Missing Scaler : {company}")

    # ------------------------
    # Dataset
    # ------------------------

    if os.path.exists(data_path):

        datasets[company] = pd.read_csv(data_path)

    else:

        print(f"❌ Missing Dataset : {company}")

print("\n--------------------------------")

print("Models Loaded :", len(models))

print("Scalers Loaded :", len(scalers))

print("Datasets Loaded :", len(datasets))

print("--------------------------------")
# =====================================================
# Predict Next Price
# =====================================================

def predict_next_price(company):

    # Check if everything is loaded
    if company not in models:
        return None

    if company not in scalers:
        return None

    if company not in datasets:
        return None

    # Load objects
    model = models[company]
    scaler = scalers[company]
    df = datasets[company]

    # Need at least 60 records
    if len(df) < 60:
        return None

    # Last 60 closing prices
    last_60 = df["Close"].values[-60:]
    last_60 = last_60.reshape(-1, 1)

    # Scale
    last_60_scaled = scaler.transform(last_60)

    # Shape -> (1,60,1)
    X_test = np.array([last_60_scaled])

    # Predict
    prediction = model.predict(X_test, verbose=0)

    # Convert back to original price
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    current_price = float(df["Close"].iloc[-1])

    predicted_return = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    return {
        "Company": company,
        "Current Price": round(current_price, 2),
        "Predicted Price": round(predicted_price, 2),
        "Predicted Return (%)": round(predicted_return, 2)
    }
    # =====================================================
# Recommend Companies
# =====================================================

def recommend_companies(investment_amount,
                        investment_period,
                        expected_return):

    results = []

    for company in COMPANIES:

        prediction = predict_next_price(company)

        if prediction is None:
            continue

        predicted_return = prediction["Predicted Return (%)"]

        # Estimated Future Value
        estimated_value = investment_amount * (
            1 + predicted_return / 100
        )

        # Profit
        profit = estimated_value - investment_amount

        # Recommendation Logic
        if predicted_return >= expected_return:

            recommendation = "🟢 Strong Buy"

            score = 100

        elif predicted_return >= expected_return * 0.70:

            recommendation = "🟢 Buy"

            score = 80

        elif predicted_return >= 0:

            recommendation = "🟡 Hold"

            score = 60

        else:

            recommendation = "🔴 Avoid"

            score = 30

        prediction["Estimated Value (₹)"] = round(
            estimated_value,
            2
        )

        prediction["Expected Profit (₹)"] = round(
            profit,
            2
        )

        prediction["Recommendation"] = recommendation

        prediction["Score"] = score

        results.append(prediction)

    recommendation_df = pd.DataFrame(results)

    recommendation_df = recommendation_df.sort_values(
        by="Predicted Return (%)",
        ascending=False
    )

    recommendation_df.reset_index(
        drop=True,
        inplace=True
    )
    return recommendation_df
    