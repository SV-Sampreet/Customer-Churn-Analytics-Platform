import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ==========================================================
# LOAD CSS
# ==========================================================

css_path = Path(__file__).resolve().parent.parent / "style.css"

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#7C3AED,#4F46E5);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:30px;
box-shadow:0 10px 30px rgba(79,70,229,.35);
">

<h1 style="margin:0;font-size:40px;">
🤖 Customer Churn Prediction
</h1>

<p style="font-size:20px;margin-top:12px;">
AI-Powered Customer Risk Prediction Dashboard
</p>

<p style="font-size:16px;color:#E5E7EB;margin-top:18px;">
Predict customer churn probability using Machine Learning models and identify high-risk customers before they leave.
</p>

<hr style="border:1px solid rgba(255,255,255,.25);">

<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:25px;">

<div>
<b>🧠 Model</b><br>
XGBoost Classifier
</div>

<div>
<b>📈 Output</b><br>
Churn Probability
</div>

<div>
<b>🎯 Purpose</b><br>
Customer Retention
</div>

<div>
<b>⚡ Prediction</b><br>
Real-Time Analysis
</div>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================================
# CUSTOMER DETAILS
# ==========================================================

st.header("📝 Customer Information")

left, right = st.columns(2)

with left:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with right:

    protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        24
    )

    monthly = st.slider(
        "Monthly Charges",
        18.0,
        120.0,
        70.0
    )

    total = st.number_input(
        "Total Charges",
        value=1700.0,
        step=10.0
    )

st.divider()

predict = st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

if predict:

    features = pd.DataFrame({
        "SeniorCitizen":[senior],
        "tenure":[tenure],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total],

        "gender_Male":[1 if gender=="Male" else 0],

        "Partner_Yes":[1 if partner=="Yes" else 0],

        "Dependents_Yes":[1 if dependents=="Yes" else 0],

        "PhoneService_Yes":[1 if phone=="Yes" else 0],

        "MultipleLines_No phone service":[
            1 if multiple=="No phone service" else 0
        ],

        "MultipleLines_Yes":[
            1 if multiple=="Yes" else 0
        ],

        "InternetService_Fiber optic":[
            1 if internet=="Fiber optic" else 0
        ],

        "InternetService_No":[
            1 if internet=="No" else 0
        ],

        "OnlineSecurity_No internet service":[
            1 if security=="No internet service" else 0
        ],

        "OnlineSecurity_Yes":[
            1 if security=="Yes" else 0
        ],

        "OnlineBackup_No internet service":[
            1 if backup=="No internet service" else 0
        ],

        "OnlineBackup_Yes":[
            1 if backup=="Yes" else 0
        ],

        "DeviceProtection_No internet service":[
            1 if protection=="No internet service" else 0
        ],

        "DeviceProtection_Yes":[
            1 if protection=="Yes" else 0
        ],

        "TechSupport_No internet service":[
            1 if support=="No internet service" else 0
        ],

        "TechSupport_Yes":[
            1 if support=="Yes" else 0
        ],

        "StreamingTV_No internet service":[
            1 if streaming_tv=="No internet service" else 0
        ],

        "StreamingTV_Yes":[
            1 if streaming_tv=="Yes" else 0
        ],

        "StreamingMovies_No internet service":[
            1 if streaming_movies=="No internet service" else 0
        ],

        "StreamingMovies_Yes":[
            1 if streaming_movies=="Yes" else 0
        ],

        "Contract_One year":[
            1 if contract=="One year" else 0
        ],

        "Contract_Two year":[
            1 if contract=="Two year" else 0
        ],

        "PaperlessBilling_Yes":[
            1 if paperless=="Yes" else 0
        ],

        "PaymentMethod_Credit card (automatic)":[
            1 if payment=="Credit card (automatic)" else 0
        ],

        "PaymentMethod_Electronic check":[
            1 if payment=="Electronic check" else 0
        ],

        "PaymentMethod_Mailed check":[
            1 if payment=="Mailed check" else 0
        ]

    })

    # ==========================================================
# MODEL PREDICTION
# ==========================================================

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    probability_percent = probability * 100

    st.divider()

    st.header("🎯 Prediction Result")

    if prediction == 1:

        st.error(
            f"""
            ## 🔴 High Churn Risk

            **Probability : {probability_percent:.2f}%**
            """
        )

    else:

        st.success(
            f"""
            ## 🟢 Customer Likely to Stay

            **Probability : {probability_percent:.2f}%**
            """
        )

    # Risk Level

    if probability_percent >= 80:
        risk = "🔴 Very High"
    elif probability_percent >= 60:
        risk = "🟠 High"
    elif probability_percent >= 40:
        risk = "🟡 Medium"
    else:
        risk = "🟢 Low"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction",
        "Churn" if prediction == 1 else "Stay"
    )

    col2.metric(
        "Probability",
        f"{probability_percent:.2f}%"
    )

    col3.metric(
        "Risk Level",
        risk
    )

    # ==========================================================
    # RETENTION CAMPAIGN OPTIMIZER
    # ==========================================================

    st.divider()

    st.header("🎯 Retention Campaign Optimizer")

    st.caption(
        "Generate a personalized customer retention strategy based on churn probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        campaign = st.selectbox(
            "Retention Campaign",
            [
                "Phone Call",
                "Email Campaign",
                "Discount Offer",
                "Premium Support",
                "Loyalty Rewards"
            ]
        )

        discount = st.slider(
            "Discount (%)",
            0,
            30,
            10
        )

    with col2:

        contact = st.selectbox(
            "Preferred Contact",
            [
                "Phone",
                "Email",
                "SMS"
            ]
        )

        budget = st.number_input(
            "Campaign Budget ($)",
            value=5000,
            step=500
        )

    # ==========================================================
    # CAMPAIGN LOGIC
    # ==========================================================

    if probability >= 0.80:

        priority = "🔴 Critical"

        expected_roi = 6.2

        recommendation = """
• Immediate phone call within 24 hours

• Offer 20% retention discount

• Assign Premium Support

• Recommend Annual Contract
"""

    elif probability >= 0.60:

        priority = "🟠 High"

        expected_roi = 4.5

        recommendation = """
• Personalized email campaign

• Offer 10% discount

• Encourage AutoPay

• Follow up within 3 days
"""

    elif probability >= 0.40:

        priority = "🟡 Medium"

        expected_roi = 2.8

        recommendation = """
• Loyalty rewards

• Educational emails

• Upsell annual plans
"""

    else:

        priority = "🟢 Low"

        expected_roi = 1.4

        recommendation = """
• Regular engagement

• Monthly newsletter

• Satisfaction survey
"""

    # ==========================================================
    # KPI CARDS
    # ==========================================================

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Priority",
        priority
    )

    k2.metric(
        "Campaign",
        campaign
    )

    k3.metric(
        "Expected ROI",
        f"{expected_roi:.1f}x"
    )

    k4.metric(
        "Budget",
        f"${budget:,.0f}"
    )

    # ==========================================================
    # RECOMMENDATION
    # ==========================================================

    st.success(f"""
## 🤖 AI Retention Recommendation

**Risk Level:** {priority}

### Recommended Actions

{recommendation}

### Campaign Configuration

• Campaign Type: **{campaign}**

• Discount: **{discount}%**

• Contact Method: **{contact}**

• Expected ROI: **{expected_roi:.1f}x**

### Expected Outcome

✔ Higher customer retention

✔ Reduced churn probability

✔ Improved customer lifetime value

✔ Increased recurring revenue
""")

    # ==========================================================
    # DOWNLOAD PLAN
    # ==========================================================

    plan = f"""Customer Churn Probability,{probability*100:.2f}%
Priority,{priority}
Campaign,{campaign}
Discount,{discount}%
Contact,{contact}
Budget,{budget}
Expected ROI,{expected_roi:.1f}x
"""

    st.download_button(
        "📥 Download Retention Plan",
        data=plan,
        file_name="retention_campaign_plan.csv",
        mime="text/csv"
    )