import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------- Page config & CSS ----------------
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header-features {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e67e22;
        margin-top: 2rem;
    }
    .sub-header-how {
        font-size: 1.5rem;
        font-weight: bold;
        color: #9b59b6;
        margin-top: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .approved-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .rejected-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Load model & preprocessor ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model_xgb.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please make sure best_model_xgb.pkl and preprocessor.pkl exist.")
        st.stop()


model, preprocessor = load_model()


# ---------------- Helper functions ----------------
def calculate_interest_rate(purpose):
    """
    Yearly interest rate (0–1) based on loan purpose.
    Rough typical ranges for personal loans / mortgages.
    """
    p = purpose.lower()

    if p == "debt consolidation":
        return 0.16      # ~16%
    elif p == "refinance":
        return 0.13      # ~13%
    elif p == "major purchase":
        return 0.14      # ~14%
    elif p == "medical":
        return 0.18      # ~18%
    elif p == "revolving credit":
        return 0.22      # ~22%
    elif p == "home improvement":
        return 0.12      # ~12%
    elif p == "home purchase":
        return 0.06      # ~6%
    else:  # "Other"
        return 0.20      # ~20%


def monthly_payment_simple(principal, annual_rate, months):
    """Simple interest: (principal + interest) / months."""
    if months <= 0:
        return 0.0
    total = principal * (1 + annual_rate)
    return total / months


def calculate_credit_score(probability):
    score = 300 + (1 - probability) * 550
    return int(np.clip(score, 300, 850))


def age_to_category(age):
    if age < 25:
        return "<25"
    elif age < 35:
        return "25-35"
    elif age < 50:
        return "35-50"
    else:
        return "50+"


def build_feature_row(
    age, monthly_income, loan_amount, loan_term, purpose, marital_status,
    available_credit, employment_length, employment_type, education,
    num_delinquencies, credit_utilization,
):
    interest_rate = calculate_interest_rate(purpose)
    monthly_payment = monthly_payment_simple(loan_amount, interest_rate, loan_term)
    payment_to_income_ratio = monthly_payment / (monthly_income + 1e-6)
    debt_service_ratio = monthly_payment / (monthly_income + 1e-6)
    available_credit_ratio = available_credit / (available_credit + loan_amount + 1e-6)
    interest_burden = interest_rate * loan_amount / (monthly_income + 1e-6)
    age_category = age_to_category(age)

    row = {
        "monthly_income": monthly_income,
        "employment_length": employment_length,
        "monthly_payment": monthly_payment,
        "payment_to_income_ratio": payment_to_income_ratio,
        "credit_utilization": credit_utilization,
        "available_credit": available_credit,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "num_delinquencies_2yrs": num_delinquencies,
        "age": age,
        "debt_service_ratio": debt_service_ratio,
        "available_credit_ratio": available_credit_ratio,
        "interest_burden": interest_burden,
        "loan_purpose": purpose,
        "employment_type": employment_type,
        "education": education,
        "marital_status": marital_status,
        "age_category": age_category,
    }
    return pd.DataFrame([row]), {
        "monthly_payment": monthly_payment,
        "interest_rate": interest_rate,
        "payment_to_income_ratio": payment_to_income_ratio,
        "debt_service_ratio": debt_service_ratio,
        "available_credit_ratio": available_credit_ratio,
        "interest_burden": interest_burden,
    }


def predict_default_probability(feature_df):
    X_proc = preprocessor.transform(feature_df)
    prob = model.predict_proba(X_proc)[0, 1]
    return prob


def apply_business_rules(default_prob, extra, model_threshold=0.31):
    """
    Business rules:
    - If payment_to_income_ratio and debt_service_ratio < 0.8 -> approve.
    - If payment_to_income_ratio or debt_service_ratio > 1.0 -> reject.
    - Otherwise use model threshold.
    """
    ratio = extra["payment_to_income_ratio"]
    ds = extra["debt_service_ratio"]

    model_pred_default = int(default_prob >= model_threshold)

    if ratio < 0.8 and ds < 0.8:
        pred_default = 0
        risk_level = "Low"
        recommendation = "APPROVE ✅"
    elif ratio > 1.0 or ds > 1.0:
        pred_default = 1
        risk_level = "High"
        recommendation = "REJECT ❌"
    else:
        pred_default = model_pred_default
        risk_level = "High" if default_prob > 0.5 else ("Medium" if default_prob > 0.3 else "Low")
        recommendation = "APPROVE ✅" if pred_default == 0 else "REJECT ❌"

    return pred_default, risk_level, recommendation


# ---------------- UI pages ----------------
def show_home():
    st.markdown('<div class="main-header">💳 Credit Score Prediction System</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header-features">🚀 Features</div>', unsafe_allow_html=True)
    st.markdown("""
- **Instant Credit Score Calculation**: Get credit scores (300–850) in seconds  
- **Default Risk Assessment**: Probability‑based risk evaluation    
- **Detailed Analytics**: Visual representations and insights for each prediction  
- **Batch Processing**: Upload CSV files for multiple predictions
""")

    st.markdown('<div class="sub-header-how">🔍 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
1. Enter customer information in the prediction page  
2. The AI model and business rules analyze the data  
3. You get an **APPROVED** or **REJECTED** decision  
4. Review detailed probability and feature analysis
""")


def show_single_prediction():
    st.markdown('<div class="main-header">💳 Credit Score Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🔮 Single Customer Prediction</div>', unsafe_allow_html=True)

    with st.form("single_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Personal information")
            age = st.number_input("Age (years)", min_value=18, max_value=80, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])  # kept for UI, not used in rate
            marital_status = st.selectbox("Marital status", ["Married", "Single", "Divorced"])
            education = st.selectbox(
                "Education level",
                ["High School", "Some College", "Bachelor", "Graduate", "Advanced"],
            )
            employment_type = st.selectbox(
                "Employment type",
                ["full-time", "part-time", "self-employed", "contract"],
            )
            employment_length = st.slider("Employment length (years)", 0, 40, 5)

            st.markdown("#### Income and loan (in USD)")
            monthly_income = st.number_input(
                "Monthly income (USD)",
                min_value=500, max_value=50_000,
                value=3_000, step=250,
            )
            loan_amount = st.number_input(
                "Loan amount (USD)",
                min_value=1_000, max_value=300_000,
                value=20_000, step=1_000,
            )
            loan_term = st.number_input(
                "Loan term (months)",
                min_value=3, max_value=120, value=36, step=1,
            )

        with col2:
            st.markdown("#### Loan details")
            purpose = st.selectbox(
                "Loan purpose",
                [
                    "Debt Consolidation",
                    "Refinance",
                    "Major Purchase",
                    "Medical",
                    "Revolving Credit",
                    "Home Improvement",
                    "Home Purchase",
                    "Other",
                ],
            )
            available_credit = st.number_input(
                "Available credit (USD)",
                min_value=0, max_value=500_000,
                value=10_000, step=1_000,
            )
            credit_utilization_pct = st.slider(
                "Credit utilization (%)",
                0.0, 100.0, 30.0, 1.0,
            )
            num_delinquencies = st.number_input(
                "Number of delinquencies (last 2 years)",
                min_value=0, max_value=10, value=0, step=1,
            )

        submitted = st.form_submit_button("🎯 Predict", use_container_width=True)

    if not submitted:
        return

    credit_utilization = credit_utilization_pct / 100.0

    feature_df, extra = build_feature_row(
        age=age,
        monthly_income=monthly_income,
        loan_amount=loan_amount,
        loan_term=loan_term,
        purpose=purpose,
        marital_status=marital_status,
        available_credit=available_credit,
        employment_length=employment_length,
        employment_type=employment_type,
        education=education,
        num_delinquencies=num_delinquencies,
        credit_utilization=credit_utilization,
    )

    with st.spinner("Calculating prediction..."):
        try:
            default_prob = predict_default_probability(feature_df)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    credit_score = calculate_credit_score(default_prob)
    pred_default, risk_level, recommendation = apply_business_rules(default_prob, extra)

    st.markdown("---")
    st.markdown('<div class="sub-header">📊 Prediction results</div>', unsafe_allow_html=True)

    if "APPROVE" in recommendation:
        banner_class = "approved-banner"
        banner_text = "APPROVED"
    else:
        banner_class = "rejected-banner"
        banner_text = "REJECTED"

    st.markdown(f'<div class="{banner_class}">{banner_text}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Default probability (model)", f"{default_prob:.2%}")
    with c2:
        st.metric("Prediction", "DEFAULT" if pred_default else "NO DEFAULT")
    with c3:
        st.metric("Risk level", risk_level)

    st.markdown("### 🔢 Derived features (USD)")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.write(f"**Monthly payment:** {extra['monthly_payment']:,.2f} USD")
    with r2:
        st.write(f"**Yearly interest rate:** {extra['interest_rate']*100:.1f}%")
    with r3:
        st.write(f"**Payment / income:** {extra['payment_to_income_ratio']:.2f}")

    r4, r5, r6 = st.columns(3)
    with r4:
        st.write(f"**Debt service ratio:** {extra['debt_service_ratio']:.2f}")
    with r5:
        st.write(f"**Available credit ratio:** {extra['available_credit_ratio']:.2f}")
    with r6:
        st.write(f"**Interest burden:** {extra['interest_burden']:.2f}")

    st.markdown("### 📈 Visualizations")
    g1, g2 = st.columns(2)

    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={'text': "Credit score"},
            gauge={
                'axis': {'range': [300, 850]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [300, 600], 'color': "lightgray"},
                    {'range': [600, 650], 'color': "gray"},
                    {'range': [650, 700], 'color': "lightblue"},
                    {'range': [700, 750], 'color': "lightgreen"},
                    {'range': [750, 850], 'color': "green"},
                ],
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        df_probs = pd.DataFrame({
            "Class": ["Default", "No default"],
            "Probability": [default_prob, 1 - default_prob],
        })
        fig2 = px.bar(
            df_probs,
            x="Class",
            y="Probability",
            color="Probability",
            range_y=[0, 1],
            color_continuous_scale="RdYlGn_r",
            title="Default vs no‑default probability",
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def show_batch_predictions():
    st.markdown('<div class="main-header">💳 Credit Score Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">📈 Batch predictions (CSV)</div>', unsafe_allow_html=True)

    st.info("""
Upload a CSV file with one row per customer.  
Required columns (amounts in **USD**, credit_utilization from 0 to 100):

- age  
- monthly_income  
- loan_amount  
- loan_term  
- purpose (Debt Consolidation, Refinance, Major Purchase, Medical,
           Revolving Credit, Home Improvement, Home Purchase, Other)  
- marital_status  
- available_credit  
- employment_length  
- employment_type  
- education  
- num_delinquencies_2yrs  
- credit_utilization (percentage, 0–100)  
- gender (Male/Female)
""")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is None:
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    st.success(f"File loaded: {len(raw_df)} rows")
    st.markdown("### Preview")
    st.dataframe(raw_df.head())

    if not st.button("🎯 Run predictions", use_container_width=True):
        return

    results = []
    progress = st.progress(0)
    status = st.empty()
    n = len(raw_df)

    for i, (_, row) in enumerate(raw_df.iterrows()):
        try:
            age = row["age"]
            monthly_income = row["monthly_income"]
            loan_amount = row["loan_amount"]
            loan_term = row["loan_term"]
            purpose = row["purpose"]
            marital_status = row["marital_status"]
            available_credit = row["available_credit"]
            employment_length = row["employment_length"]
            employment_type = row["employment_type"]
            education = row["education"]
            num_delinquencies = row["num_delinquencies_2yrs"]
            credit_utilization = row["credit_utilization"] / 100.0
            gender = row["gender"]  # not used in calc but kept for schema

            feature_df, extra = build_feature_row(
                age=age,
                monthly_income=monthly_income,
                loan_amount=loan_amount,
                loan_term=loan_term,
                purpose=purpose,
                marital_status=marital_status,
                available_credit=available_credit,
                employment_length=employment_length,
                employment_type=employment_type,
                education=education,
                num_delinquencies=num_delinquencies,
                credit_utilization=credit_utilization,
            )
            prob = predict_default_probability(feature_df)
            score = calculate_credit_score(prob)
            pred_default, risk_level, recommendation = apply_business_rules(prob, extra)

            results.append({
                "row_id": i + 1,
                "default_probability": prob,
                "credit_score": score,
                "prediction": "DEFAULT" if pred_default else "NO DEFAULT",
                "risk_level": risk_level,
                "recommendation": recommendation,
                "monthly_payment_usd": extra["monthly_payment"],
                "interest_rate_yearly": extra["interest_rate"],
                "payment_to_income_ratio": extra["payment_to_income_ratio"],
                "debt_service_ratio": extra["debt_service_ratio"],
                "available_credit_ratio": extra["available_credit_ratio"],
                "interest_burden": extra["interest_burden"],
            })
        except Exception as e:
            results.append({
                "row_id": i + 1,
                "default_probability": np.nan,
                "credit_score": "ERROR",
                "prediction": f"ERROR: {str(e)[:40]}",
                "risk_level": "ERROR",
                "recommendation": "ERROR",
                "monthly_payment_usd": np.nan,
                "interest_rate_yearly": np.nan,
                "payment_to_income_ratio": np.nan,
                "debt_service_ratio": np.nan,
                "available_credit_ratio": np.nan,
                "interest_burden": np.nan,
            })

        progress.progress((i + 1) / n)
        status.text(f"Processed {i + 1} / {n}")

    progress.empty()
    status.empty()

    res_df = pd.DataFrame(results)
    st.markdown("### Results")
    st.dataframe(res_df)

    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download predictions as CSV",
        data=csv_bytes,
        file_name=f"credit_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------- Main routing ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select page",
    ["🏠 Home", "🔮 Single prediction", "📈 Batch predictions"],
)

if page == "🏠 Home":
    show_home()
elif page == "🔮 Single prediction":
    show_single_prediction()
else:
    show_batch_predictions()