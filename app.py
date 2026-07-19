import pickle
import pandas as pd
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Risk Prediction")
st.write(
    "Predict whether a borrower is at risk of loan default based on financial and repayment history."
)

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------


# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    rev_util = st.number_input("Revolving Credit Utilization", min_value=0.0, value=0.50)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    late_30_59 = st.number_input("30-59 Days Past Due", min_value=0, value=0)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.40)
    monthly_inc = st.number_input("Monthly Income", min_value=0.0, value=50000.0)

with col2:
    open_credit = st.number_input("Open Credit Lines", min_value=0, value=5)
    late_90 = st.number_input("90+ Days Past Due", min_value=0, value=0)
    real_estate = st.number_input("Real Estate Loans", min_value=0, value=1)
    late_60_89 = st.number_input("60-89 Days Past Due", min_value=0, value=0)
    dependents = st.number_input("Dependents", min_value=0, value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk"):

    input_df = pd.DataFrame([{
        "rev_util": rev_util,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("### Prediction: High Risk Borrower")
    else:
        st.success("### Prediction: Low Risk Borrower")

    st.metric(
        "Probability of Default",
        f"{probability*100:.2f}%"
    )

