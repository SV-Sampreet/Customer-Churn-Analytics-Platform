import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# LOAD CSS
# ==========================================================

css_path = Path(__file__).resolve().parent.parent / "style.css"

with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/telco_churn_feature_engineered.csv"
    )

df = load_data()

X = df.drop(columns=["Churn"])
# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/xgboost.pkl")

# ==========================================================
# SHAP EXPLAINER
# ==========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#7C3AED,#4C1D95);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:30px;
box-shadow:0 10px 30px rgba(124,58,237,.35);
">

<h1 style="margin:0;font-size:40px;">
🧠 SHAP Explainability Dashboard
</h1>

<p style="font-size:20px;margin-top:12px;">
Explainable AI for Customer Churn Prediction
</p>

<p style="font-size:16px;color:#E5E7EB;margin-top:18px;">
Understand how the Machine Learning model makes predictions using SHAP values. Discover the most influential features, explain individual customer predictions, and gain actionable business insights through Explainable AI.
</p>

<hr style="border:1px solid rgba(255,255,255,.25);">

<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:25px;">

<div>
<b>🧠 Explainable AI</b><br>
SHAP Values
</div>

<div>
<b>⭐ Insights</b><br>
Feature Importance
</div>

<div>
<b>🎯 Analysis</b><br>
Customer-Level Explanation
</div>

<div>
<b>💼 Business Value</b><br>
Transparent AI Decisions
</div>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# SHAP SUMMARY PLOT
# ==========================================================

st.subheader("📊 SHAP Summary Plot")

fig, ax = plt.subplots(figsize=(12,6))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

st.pyplot(fig)

plt.close()

st.divider()




# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.subheader("⭐ Global Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


# ==========================================================
# AI EXECUTIVE COPILOT
# ==========================================================

st.divider()

st.subheader("🤖 AI Executive Copilot")

st.caption(
    "Business interpretation of the model's most important features."
)

top5 = importance.head(5)["Feature"].tolist()

recommendations = []

if "Contract_Two year" in top5:
    recommendations.append("✔ Promote long-term contracts to improve customer retention.")

if "MonthlyCharges" in top5:
    recommendations.append("✔ Offer personalized discounts for customers with high monthly charges.")

if "PaymentMethod_Electronic check" in top5:
    recommendations.append("✔ Encourage customers to switch to AutoPay.")

if "tenure" in top5:
    recommendations.append("✔ Reward long-tenure customers through loyalty programs.")

if "InternetService_Fiber optic" in top5:
    recommendations.append("✔ Improve Fiber Optic customer experience.")

if len(recommendations) == 0:
    recommendations.append("✔ Continue monitoring customer behaviour.")

left, right = st.columns(2)

with left:

    st.info(f"""
### 🔍 Key Business Drivers

• {top5[0]}

• {top5[1]}

• {top5[2]}

• {top5[3]}

• {top5[4]}
""")

with right:

    st.success(f"""
### 📋 AI Executive Recommendations

{chr(10).join(recommendations)}

---

### Expected Business Benefits

✔ Reduce customer churn

✔ Improve customer lifetime value

✔ Increase recurring revenue

✔ Strengthen customer loyalty
""")


fig = px.bar(
    importance.head(20),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    text_auto=".3f",
    title="Top 20 Important Features"
)

fig.update_layout(
    height=700
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# TOP 10 FEATURES TABLE
# ==========================================================

st.subheader("📋 Top 10 Most Influential Features")

st.dataframe(
    importance.head(10),
    use_container_width=True,
    height=420
)

st.divider()

# ==========================================================
# GLOBAL INSIGHTS
# ==========================================================

st.subheader("🌍 Global Model Insights")

left,right = st.columns(2)

with left:

    st.success(f"""
### Positive Drivers

• {importance.iloc[0]['Feature']}

• {importance.iloc[1]['Feature']}

• {importance.iloc[2]['Feature']}

• {importance.iloc[3]['Feature']}

• {importance.iloc[4]['Feature']}
""")

with right:

    st.info("""
### Business Interpretation

✅ These features contribute the most
towards customer churn predictions.

Business teams should continuously
monitor these attributes and create
targeted retention strategies.

The higher the SHAP value, the greater
its impact on the model prediction.
""")

st.divider()

# ==========================================================
# SELECT CUSTOMER
# ==========================================================

st.subheader("🎯 Explain Individual Customer")

customer_id = st.slider(
    "Select Customer Index",
    0,
    len(X)-1,
    0
)

st.dataframe(
    X.iloc[[customer_id]],
    use_container_width=True
)

st.divider()

# ==========================================================
# SHAP WATERFALL
# ==========================================================

st.subheader("🌊 SHAP Waterfall Plot")

fig, ax = plt.subplots(figsize=(12,7))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[customer_id],
        base_values=explainer.expected_value,
        data=X.iloc[customer_id],
        feature_names=X.columns
    ),
    show=False
)

st.pyplot(fig)

plt.close()

st.divider()

# ==========================================================
# TOP FEATURES FOR CUSTOMER
# ==========================================================

st.subheader("⭐ Top Feature Contributions")

customer_shap = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Value": shap_values[customer_id]
})

customer_shap["Absolute"] = customer_shap["SHAP Value"].abs()

customer_shap = customer_shap.sort_values(
    "Absolute",
    ascending=False
)

fig = px.bar(
    customer_shap.head(15),
    x="SHAP Value",
    y="Feature",
    orientation="h",
    color="SHAP Value",
    title="Top Feature Contributions"
)

fig.update_layout(height=650)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# FEATURE CONTRIBUTION TABLE
# ==========================================================

st.subheader("📋 Feature Contribution Table")

st.dataframe(
    customer_shap.head(20),
    use_container_width=True,
    height=450
)

st.divider()

# ==========================================================
# BUSINESS EXPLANATION
# ==========================================================

st.subheader("💼 Business Interpretation")

top5 = customer_shap.head(5)["Feature"].tolist()

st.info(f"""
### Why was this prediction made?

The prediction was mainly influenced by:

• {top5[0]}

• {top5[1]}

• {top5[2]}

• {top5[3]}

• {top5[4]}

Higher positive SHAP values increase churn probability.

Negative SHAP values reduce churn probability.

This explanation helps business teams understand exactly
why the model predicted churn for this customer.
""")

st.divider()

# ==========================================================
# EXECUTIVE RECOMMENDATIONS
# ==========================================================

st.subheader("🤖 AI Executive Recommendations")

st.success("""
### Recommended Actions

✅ Target customers with Month-to-Month contracts.

✅ Encourage migration to One-Year or Two-Year contracts.

✅ Promote AutoPay over Electronic Check.

✅ Offer loyalty rewards for long-tenure customers.

✅ Prioritize customers with high monthly charges.

✅ Improve Tech Support for high-risk customers.

✅ Launch personalized retention campaigns.

✅ Continuously monitor SHAP explanations for changing trends.
""")

st.divider()

