import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictor import recommend_companies
# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Investment Company Recommendation System",
    page_icon="📈",
    layout="wide")
# CUSTOM CSS
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.block-container{
    padding-top:2rem;
}

div[data-testid="stMetric"]{
    background:white;
    padding:18px;
    border-radius:12px;
    border:1px solid #dddddd;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

h1{
    color:#0F172A;
}

h2{
    color:#1E40AF;
}

h3{
    color:#2563EB;
}

</style>
""", unsafe_allow_html=True)
# TITLE
st.title("📈 AI Investment Company Recommendation System")
st.markdown("""
### Deep Learning Based Stock Price Prediction & Recommendation

This system predicts future stock prices using **LSTM Deep Learning**
and recommends the best investment companies based on:

- 💰 Investment Amount
- 📅 Investment Period
- 🎯 Expected Return

""")

st.markdown("---")
# SIDEBAR
st.sidebar.title("Investment Details")
investment_amount = st.sidebar.number_input(
    "Investment Amount (₹)",
    min_value=1000,
    max_value=10000000,
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
recommend_button = st.sidebar.button("🚀 Recommend Companies")
# GENERATE RECOMMENDATION
if recommend_button:
    with st.spinner("Analyzing all companies..."):
        recommendation_df = recommend_companies(
            investment_amount=investment_amount,
            investment_period=investment_period,
            expected_return=expected_return
        )
    if recommendation_df.empty:
        st.error("No recommendations could be generated.")
    else:
        st.success("Recommendation Generated Successfully!")
        # Sort recommendations by predicted return
recommendation_df = recommendation_df.sort_values(
    by="Predicted Return (%)",
    ascending=False
).reset_index(drop=True)

# Select the best company
best_company = recommendation_df.iloc[0]
st.markdown("---")
st.subheader("🏆 Best Investment Recommendation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🏢 Company",
        best_company["Company"]
    )

with col2:
    st.metric(
        "💹 Current Price",
        f"₹{best_company['Current Price']:,.2f}"
    )

with col3:
    st.metric(
        "📈 Predicted Price",
        f"₹{best_company['Predicted Price']:,.2f}"
    )

with col4:
    st.metric(
        "📊 Predicted Return",
        f"{best_company['Predicted Return (%)']:.2f}%"
    )

col5, col6, col7 = st.columns(3)

with col5:
    st.metric(
        "💰 Estimated Value",
        f"₹{best_company['Estimated Value (₹)']:,.2f}"
    )

with col6:
    st.metric(
        "💵 Expected Profit",
        f"₹{best_company['Expected Profit (₹)']:,.2f}"
    )

with col7:
    st.metric(
        "⭐ Recommendation",
        best_company["Recommendation"]
    )

st.markdown("---")
st.subheader("🏆 Top 5 Recommended Companies")

top5 = recommendation_df.head(5)

st.dataframe(
    top5,
    use_container_width=True
)
st.subheader("📋 Complete Recommendation Table")
st.dataframe(
            recommendation_df,
            use_container_width=True
        )
st.markdown("---")
import plotly.express as px

st.markdown("---")
st.subheader("📊 Predicted Return Comparison")

chart_df = recommendation_df.copy()

fig = px.bar(
    chart_df,
    x="Company",
    y="Predicted Return (%)",
    text="Predicted Return (%)",
    color="Predicted Return (%)",
    color_continuous_scale="Viridis",
    title="Predicted Return (%) of All Companies"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Company",
    yaxis_title="Predicted Return (%)",
    height=550,
    showlegend=False,
    title_x=0.5
)
st.markdown("---")
st.subheader("📈 Project Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Companies Analysed", len(recommendation_df))

with col2:
    st.metric(
        "Highest Return",
        f"{recommendation_df['Predicted Return (%)'].max():.2f}%"
    )

with col3:
    st.metric(
        "Average Return",
        f"{recommendation_df['Predicted Return (%)'].mean():.2f}%"
    )

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.markdown("---")

st.subheader("💼 Investment Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Investment Amount")
    st.metric(
        label="",
        value=f"₹{investment_amount:,.0f}"
    )

with col2:
    st.markdown("#### Investment Period")
    st.metric(
        label="",
        value=f"{investment_period} Years"
    )

with col3:
    st.markdown("#### Expected Return")
    st.metric(
        label="",
        value=f"₹{best_company['Expected Profit (₹)']:,.2f}"
    )

csv = recommendation_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Recommendations",
    data=csv,
    file_name="Investment_Recommendations.csv",
    mime="text/csv"
)
st.subheader("📈 Predicted Return Comparison")

chart_df = recommendation_df.set_index("Company")

st.bar_chart(chart_df["Predicted Return (%)"])
st.subheader("💼 Investment Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Investment Amount",
        f"₹{investment_amount:,.0f}"
    )

with col2:
    st.metric(
        "Investment Period",
        f"{investment_period} Years"
    )

with col3:
    st.metric(
        "Expected Return",
        f"{expected_return}%"
    )
    st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;'>

    ## 📈 AI Investment Company Recommendation System

    **Deep Learning Based Stock Price Prediction using LSTM**

    ### 🛠️ Technologies Used

    Python • TensorFlow • Keras • LSTM • Pandas • NumPy • Scikit-Learn • Streamlit • Plotly • yFinance

    ---

    👩‍💻 **Developed By:** Vaibhavi Mahadik

    🎓 Final Year Deep Learning Project

    📅 Academic Year: 2026

    </div>
    """,
    unsafe_allow_html=True
)