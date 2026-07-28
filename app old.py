import streamlit as st
import matplotlib.pyplot as plt

from predictor import recommend_companies

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Investment Company Recommendation System",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("📈 AI Investment Company Recommendation System")
st.markdown("### Deep Learning Based Stock Recommendation")

st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Investment Details")

investment_amount = st.sidebar.number_input(
    "Investment Amount (₹)",
    min_value=1000,
    value=100000,
    step=1000
)

investment_period = st.sidebar.slider(
    "Investment Period (Years)",
    min_value=1,
    max_value=20,
    value=5
)

expected_return = st.sidebar.slider(
    "Expected Return (%)",
    min_value=1,
    max_value=30,
    value=10
)

recommend = st.sidebar.button("Recommend Companies")

# -------------------------------
# Recommendation
# -------------------------------
if recommend:

    with st.spinner("Analyzing Companies..."):

        recommendation = recommend_companies(
            investment_amount,
            investment_period,
            expected_return
        )

    if recommendation.empty:

        st.error("No recommendation could be generated.")

    else:

        best = recommendation.iloc[0]

        st.success("Recommendation Generated Successfully!")

        st.subheader("🏆 Best Company Recommendation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Company",
                best["Company"]
            )

            st.metric(
                "Recommendation",
                best["Recommendation"]
            )

        with col2:
            st.metric(
                "Current Price",
                f"₹ {best['Current Price']}"
            )

            st.metric(
                "Predicted Price",
                f"₹ {best['Predicted Price']}"
            )

        with col3:
            st.metric(
                "Predicted Return",
                f"{best['Predicted Return (%)']} %"
            )

            st.metric(
                "Estimated Value",
                f"₹ {best['Estimated Value (₹)']:,}"
            )

        st.markdown("---")

        st.subheader("📋 Top 5 Recommended Companies")

        top5 = recommendation.head(5)

        st.dataframe(
            top5,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 All Company Rankings")

        st.dataframe(
            recommendation,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📈 Predicted Return Comparison")

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(
            recommendation["Company"],
            recommendation["Predicted Return (%)"]
        )

        plt.xticks(rotation=45)
        plt.xlabel("Company")
        plt.ylabel("Predicted Return (%)")

        st.pyplot(fig)

        st.markdown("---")

        st.subheader("💰 Your Investment")

        st.write(f"**Investment Amount:** ₹ {investment_amount:,.0f}")
        st.write(f"**Investment Period:** {investment_period} Years")
        st.write(f"**Expected Return:** {expected_return}%")