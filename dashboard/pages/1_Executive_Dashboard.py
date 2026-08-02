import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)





# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ===========================
   MAIN PAGE
=========================== */

.stApp{
    background-color:#0E1117;
}

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:1rem;
    padding-left:2rem;
    padding-right:2rem;
    padding-bottom:2rem;
}

/* ===========================
   TITLES
=========================== */

h1,h2,h3,h4,h5,h6{
    color:white !important;
    font-weight:700;
}

/* ===========================
   SIDEBAR
=========================== */

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* ===========================
   KPI CARD
=========================== */

[data-testid="stMetric"]{

    background:linear-gradient(
        135deg,
        #1E293B 0%,
        #0F172A 100%
    );

    border-radius:18px;

    padding:22px;

    border:1px solid rgba(255,255,255,.08);

    border-left:6px solid #3B82F6;

    box-shadow:
        0 10px 30px rgba(0,0,0,.35);

}

/* KPI LABEL */

[data-testid="stMetricLabel"]{

    color:#CBD5E1 !important;

    font-size:16px !important;

    font-weight:600 !important;

}

/* KPI VALUE */

[data-testid="stMetricValue"]{

    color:white !important;

    font-size:40px !important;

    font-weight:800 !important;

}

/* KPI DELTA */

[data-testid="stMetricDelta"]{

    color:#22C55E !important;

}

/* ===========================
   DATAFRAME
=========================== */

[data-testid="stDataFrame"]{

    border-radius:12px;

}

/* ===========================
   PLOTLY
=========================== */

.js-plotly-plot{

    border-radius:14px;

}

/* ===========================
   BUTTONS
=========================== */

.stButton>button{

    background:#2563EB;

    color:white;

    border-radius:8px;

    border:none;

}

.stButton>button:hover{

    background:#1D4ED8;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/telco_churn_cleaned.csv")
    return df

df = load_data()

# ==========================================================
# HEADER
# ==========================================================

left,right=st.columns([4,1])

with left:

    st.title("📊 Executive Dashboard")

    st.caption("Customer Churn Analytics Platform")

with right:

    st.metric(
        "Updated",
        datetime.now().strftime("%d %b %Y")
    )

st.divider()


# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#2563EB,#1E40AF);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:25px;
box-shadow:0 10px 30px rgba(37,99,235,.35);
">

<h1 style="margin:0;font-size:42px;">
📊 Customer Churn Analytics Platform
</h1>

<p style="font-size:20px;margin-top:12px;">
Enterprise AI Dashboard for Customer Retention & Business Intelligence
</p>

<p style="font-size:16px;color:#E2E8F0;margin-top:20px;">
✔ Executive KPIs &nbsp;&nbsp;
✔ Customer Churn Prediction &nbsp;&nbsp;
✔ Business Insights &nbsp;&nbsp;
✔ Model Performance &nbsp;&nbsp;
✔ SHAP Explainability
</p>

<hr style="border:1px solid rgba(255,255,255,.25);">

<div style="display:flex;gap:40px;font-size:15px;">

<div>
<b>🧠 Machine Learning</b><br>
XGBoost • Random Forest • Logistic Regression
</div>

<div>
<b>📈 Analytics</b><br>
Plotly • Pandas • SQL • Streamlit
</div>

<div>
<b>🎯 Objective</b><br>
Reduce Customer Churn & Improve Revenue
</div>

</div>

</div>
""", unsafe_allow_html=True)




# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=80
)

st.sidebar.title("Analytics Filters")

gender=st.sidebar.multiselect(
    "Gender",
    sorted(df["gender"].unique()),
    default=sorted(df["gender"].unique())
)

contract=st.sidebar.multiselect(
    "Contract",
    sorted(df["Contract"].unique()),
    default=sorted(df["Contract"].unique())
)

internet=st.sidebar.multiselect(
    "Internet Service",
    sorted(df["InternetService"].unique()),
    default=sorted(df["InternetService"].unique())
)

senior=st.sidebar.multiselect(
    "Senior Citizen",
    sorted(df["SeniorCitizen"].unique()),
    default=sorted(df["SeniorCitizen"].unique())
)

filtered_df=df[
    df["gender"].isin(gender)
    &
    df["Contract"].isin(contract)
    &
    df["InternetService"].isin(internet)
    &
    df["SeniorCitizen"].isin(senior)
]

st.sidebar.markdown("---")

st.sidebar.success(
    f"""
Customers Selected

{len(filtered_df):,}
"""
)

st.sidebar.info(
"""
Dashboard Version

Enterprise v1.0
"""
)

# ==========================================================
# KPI ENGINE
# ==========================================================

total_customers=len(filtered_df)

churn_customers=(
    filtered_df["Churn"]=="Yes"
).sum()

churn_rate=(
    churn_customers/total_customers
)*100

monthly_revenue=filtered_df["MonthlyCharges"].sum()

revenue_risk=filtered_df.loc[
    filtered_df["Churn"]=="Yes",
    "MonthlyCharges"
].sum()

avg_monthly=filtered_df["MonthlyCharges"].mean()

avg_tenure=filtered_df["tenure"].mean()

# ==========================================================
# KPI CARDS
# ==========================================================

st.markdown("## 📊 Executive KPIs")

st.markdown(f"""
<style>

.kpi-container{{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:20px;
margin-top:20px;
margin-bottom:20px;
}}

.kpi-card{{
background:linear-gradient(135deg,#1E293B,#0F172A);
padding:24px;
border-radius:18px;
border-left:6px solid #3B82F6;
box-shadow:0 8px 24px rgba(0,0,0,.35);
}}

.kpi-title{{
color:#CBD5E1;
font-size:15px;
font-weight:600;
}}

.kpi-value{{
color:white;
font-size:42px;
font-weight:800;
margin-top:8px;
}}

</style>

<div class="kpi-container">

<div class="kpi-card">
<div class="kpi-title">👥 Total Customers</div>
<div class="kpi-value">{total_customers:,}</div>
</div>

<div class="kpi-card">
<div class="kpi-title">🚨 Churn Customers</div>
<div class="kpi-value">{churn_customers:,}</div>
</div>

<div class="kpi-card">
<div class="kpi-title">📉 Churn Rate</div>
<div class="kpi-value">{churn_rate:.2f}%</div>
</div>

<div class="kpi-card">
<div class="kpi-title">💰 Revenue At Risk</div>
<div class="kpi-value">${revenue_risk:,.0f}</div>
</div>

<div class="kpi-card">
<div class="kpi-title">💵 Monthly Revenue</div>
<div class="kpi-value">${monthly_revenue:,.0f}</div>
</div>

<div class="kpi-card">
<div class="kpi-title">📅 Avg Tenure</div>
<div class="kpi-value">{avg_tenure:.1f}</div>
</div>

</div>

""", unsafe_allow_html=True)

st.divider()


# ==========================================================
# EXECUTIVE SUMMARY BAR
# ==========================================================

st.markdown("## 📌 Executive Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
### 💼 Business Health

- 👥 Customers : **{total_customers:,}**
- 📉 Churn Rate : **{churn_rate:.2f}%**
- 💰 Revenue : **${monthly_revenue:,.0f}**
""")

with col2:
    st.warning(f"""
### ⚠ Risk Overview

- 🚨 Churn Customers : **{churn_customers:,}**
- 💵 Revenue At Risk : **${revenue_risk:,.0f}**
- 📅 Avg Tenure : **{avg_tenure:.1f} Months**
""")

with col3:
    top_contract = filtered_df["Contract"].mode()[0]
    top_service = filtered_df["InternetService"].mode()[0]

    st.success(f"""
### 🎯 Quick Insights

- 📄 Top Contract : **{top_contract}**
- 🌐 Top Internet : **{top_service}**
- 📊 Dashboard : **Enterprise Edition**
""")

st.divider()


# ==========================================================
# CEO SCENARIO SIMULATOR
# ==========================================================

st.divider()

st.header("🚀 CEO Scenario Simulator")

st.caption(
    "Estimate how different business strategies could reduce churn and protect revenue."
)

col1, col2 = st.columns(2)

with col1:
    contract_migration = st.slider(
        "Move Month-to-Month Customers to Annual Contracts (%)",
        0,
        100,
        30
    )

    autopay_adoption = st.slider(
        "Increase AutoPay Adoption (%)",
        0,
        100,
        20
    )

with col2:
    discount = st.selectbox(
        "Retention Discount",
        [0, 5, 10, 15, 20],
        index=2
    )

    fiber_improvement = st.slider(
        "Improve Fiber Service Quality (%)",
        0,
        100,
        25
    )

# ==========================================================
# SIMPLE BUSINESS SIMULATION
# ==========================================================

current_churn = churn_rate

reduction = (
    contract_migration * 0.05 +
    autopay_adoption * 0.03 +
    discount * 0.15 +
    fiber_improvement * 0.02
)

estimated_churn = max(current_churn - reduction, 5)

customers_saved = int(
    total_customers *
    (current_churn - estimated_churn) /
    100
)

annual_revenue_saved = customers_saved * avg_monthly * 12

roi = (
    annual_revenue_saved /
    max(customers_saved * 40, 1)
)

# ==========================================================
# RESULTS
# ==========================================================

st.markdown("### 📈 Estimated Business Impact")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Churn",
    f"{current_churn:.2f}%"
)

c2.metric(
    "Estimated Churn",
    f"{estimated_churn:.2f}%",
    delta=f"-{current_churn-estimated_churn:.2f}%"
)

c3.metric(
    "Customers Retained",
    f"{customers_saved:,}"
)

c4.metric(
    "Revenue Protected",
    f"${annual_revenue_saved:,.0f}"
)

st.metric(
    "Estimated ROI",
    f"{roi:.1f}x"
)

# ==========================================================
# EXECUTIVE RECOMMENDATION
# ==========================================================

st.success(f"""
### 🤖 Executive Recommendation

Based on the selected business strategy:

• Estimated Churn Rate: **{estimated_churn:.2f}%**

• Customers Retained: **{customers_saved:,}**

• Estimated Annual Revenue Protected:
**${annual_revenue_saved:,.0f}**

### Recommended Actions

✅ Convert Month-to-Month customers to Annual Contracts

✅ Increase AutoPay adoption

✅ Improve Fiber Optic service quality

✅ Offer targeted retention discounts

> **Note:** This is a business scenario simulation based on configurable assumptions. It is intended for planning and strategy discussions rather than predicting actual intervention outcomes.
""")




# ==========================================================
# CUSTOMER ANALYTICS
# ==========================================================

st.markdown("## 📈 Customer Analytics")

col1, col2 = st.columns(2)

with col1:

    churn_df = (
        filtered_df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_df.columns = ["Churn", "Customers"]

    fig = px.pie(
        churn_df,
        names="Churn",
        values="Customers",
        hole=0.55,
        title="Customer Churn Distribution",
        color="Churn",
        color_discrete_map={
            "Yes":"#EF4444",
            "No":"#10B981"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    contract_df = (
        filtered_df.groupby(
            ["Contract","Churn"]
        )
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        contract_df,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="stack",
        title="Contract Type vs Churn"
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# SERVICE ANALYTICS
# ==========================================================

st.markdown("## 🌐 Service Analytics")

col1, col2 = st.columns(2)

with col1:

    internet_df = (
        filtered_df.groupby(
            ["InternetService","Churn"]
        )
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        internet_df,
        x="InternetService",
        y="Customers",
        color="Churn",
        barmode="stack",
        title="Internet Service vs Churn"
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    payment_df = (
        filtered_df.groupby(
            ["PaymentMethod","Churn"]
        )
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        payment_df,
        x="PaymentMethod",
        y="Customers",
        color="Churn",
        barmode="stack",
        title="Payment Method vs Churn"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-20,
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# CUSTOMER VALUE ANALYTICS
# ==========================================================

st.markdown("## 💰 Customer Value Analytics")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered_df,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution",
        color_discrete_sequence=["#2563EB"]
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        filtered_df,
        x="tenure",
        nbins=30,
        title="Customer Tenure Distribution",
        color_discrete_sequence=["#F97316"]
    )

    fig.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# REVENUE ANALYTICS
# ==========================================================

st.markdown("## 💵 Revenue Analytics")

col1, col2 = st.columns(2)

with col1:

    revenue_contract = (
        filtered_df.groupby("Contract")["MonthlyCharges"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        revenue_contract,
        x="Contract",
        y="MonthlyCharges",
        color="MonthlyCharges",
        title="Revenue by Contract",
        text_auto=".2s",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    revenue_churn = (
        filtered_df.groupby("Churn")["MonthlyCharges"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        revenue_churn,
        names="Churn",
        values="MonthlyCharges",
        hole=.55,
        title="Revenue by Churn Status",
        color="Churn",
        color_discrete_map={
            "Yes":"#DC2626",
            "No":"#16A34A"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# CUSTOMER SEGMENT ANALYTICS
# ==========================================================

st.markdown("## 🎯 Customer Segmentation")

col1, col2 = st.columns(2)

with col1:

    segment = (
        filtered_df.groupby("Contract")
        .agg(
            Customers=("customerID","count"),
            AvgMonthlyCharges=("MonthlyCharges","mean"),
            AvgTenure=("tenure","mean")
        )
        .reset_index()
    )

    fig = px.scatter(
        segment,
        x="AvgMonthlyCharges",
        y="AvgTenure",
        size="Customers",
        color="Contract",
        hover_name="Contract",
        title="Customer Segment Analysis",
        size_max=60
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    churn_contract = (
        filtered_df.groupby("Contract")["Churn"]
        .apply(lambda x:(x=="Yes").mean()*100)
        .reset_index(name="ChurnRate")
    )

    fig = px.bar(
        churn_contract,
        x="Contract",
        y="ChurnRate",
        color="ChurnRate",
        text_auto=".2f",
        title="Churn Rate by Contract",
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==========================================================
# CUSTOMER RISK TABLE
# ==========================================================

st.markdown("## 🚨 High Risk Customers")

risk_df = (
    filtered_df[
        filtered_df["Churn"]=="Yes"
    ]
    .sort_values(
        by="MonthlyCharges",
        ascending=False
    )
)

risk_df["Risk Level"] = risk_df["MonthlyCharges"].apply(
    lambda x:
    "Critical" if x>=90 else
    "High" if x>=70 else
    "Medium"
)

show_cols = [
    "customerID",
    "Contract",
    "InternetService",
    "MonthlyCharges",
    "tenure",
    "Risk Level"
]

st.dataframe(
    risk_df[show_cols].head(20),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

highest_contract = (
    filtered_df["Contract"]
    .value_counts()
    .idxmax()
)

highest_service = (
    filtered_df["InternetService"]
    .value_counts()
    .idxmax()
)

highest_payment = (
    filtered_df["PaymentMethod"]
    .value_counts()
    .idxmax()
)

highest_charge = (
    filtered_df["MonthlyCharges"]
    .max()
)

st.markdown("## 🤖 Executive Insights")

st.success(f"""

### Business Summary

👥 Customers Analysed : **{total_customers:,}**

🚨 Churn Customers : **{churn_customers:,}**

📉 Churn Rate : **{churn_rate:.2f}%**

💰 Revenue At Risk : **${revenue_risk:,.2f}**

💵 Monthly Revenue : **${monthly_revenue:,.2f}**

📅 Average Tenure : **{avg_tenure:.1f} Months**

---

### Key Insights

• Most customers use **{highest_contract}** contracts.

• Most customers use **{highest_service}** internet.

• Most common payment method is **{highest_payment}**.

• Highest monthly customer charge is **${highest_charge:.2f}**.

---

### Recommended Actions

✅ Convert Month-to-Month customers into yearly plans.

✅ Target customers paying more than $70/month.

✅ Create retention campaigns for high-value customers.

✅ Reward loyal customers with tenure greater than 24 months.

✅ Monitor this dashboard weekly.

""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.caption(
    f"""
Customer Churn Analytics Platform • Executive Dashboard

Last Updated : {datetime.now().strftime("%d %B %Y %H:%M")}

Developed using Python • Streamlit • Plotly • Machine Learning
"""
)


# ==========================================================
# BUSINESS HEALTH SCORECARD
# ==========================================================

st.markdown("## 🏆 Business Health Scorecard")

col1, col2, col3, col4 = st.columns(4)

health = max(0, min(100, round(100 - churn_rate)))

retention = 100 - churn_rate

risk_score = max(0, min(100, round((revenue_risk / monthly_revenue) * 100))) if monthly_revenue else 0

growth_score = round((avg_tenure / filtered_df["tenure"].max()) * 100) if filtered_df["tenure"].max() > 0 else 0

col1.metric(
    "Business Health",
    f"{health}/100"
)

col2.metric(
    "Customer Retention",
    f"{retention:.1f}%"
)

col3.metric(
    "Revenue Risk Score",
    f"{risk_score}%"
)

col4.metric(
    "Loyalty Score",
    f"{growth_score}%"
)

st.divider()





# ==========================================================
# ADVANCED BUSINESS INTELLIGENCE
# ==========================================================

st.markdown("## 📊 Advanced Business Intelligence")

col1, col2 = st.columns(2)

with col1:

    churn_gender = (
        filtered_df.groupby(["gender", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        churn_gender,
        x="gender",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn by Gender",
        text_auto=True,
        color_discrete_map={
            "Yes":"#EF4444",
            "No":"#10B981"
        }
    )

    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(
        color="white",
        size=14
    ),
    title_font=dict(
        size=22
    ),
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),
    height=430
)

    st.plotly_chart(fig, use_container_width=True)

with col2:

    senior_df = (
        filtered_df.groupby(["SeniorCitizen","Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        senior_df,
        x="SeniorCitizen",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Senior Citizen Churn",
        text_auto=True,
        color_discrete_map={
            "Yes":"#EF4444",
            "No":"#10B981"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=430
    )

    st.plotly_chart(fig,use_container_width=True)

st.divider()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.markdown("## 🔥 Correlation Analysis")

numeric = filtered_df.select_dtypes(include="number")

corr = numeric.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Matrix"
)

fig.update_layout(
    template="plotly_white",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# TOP REVENUE CUSTOMERS
# ==========================================================

st.markdown("## 💎 Top Revenue Customers")

top_customer = (
    filtered_df.sort_values(
        "MonthlyCharges",
        ascending=False
    )
)

cols = [
    "customerID",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "MonthlyCharges",
    "tenure"
]

st.dataframe(
    top_customer[cols].head(15),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# DATA QUALITY
# ==========================================================

st.markdown("## ✅ Dataset Quality")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Rows",
    len(filtered_df)
)

c2.metric(
    "Columns",
    filtered_df.shape[1]
)

c3.metric(
    "Missing Values",
    int(filtered_df.isnull().sum().sum())
)

c4.metric(
    "Duplicate Rows",
    int(filtered_df.duplicated().sum())
)

st.divider()

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.markdown("## 📥 Download Report")

csv = filtered_df.to_csv(index=False)

st.download_button(
    "⬇ Download Current Filtered Dataset",
    csv,
    "customer_churn_dashboard.csv",
    "text/csv"
)

st.divider()

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

with st.expander("ℹ About This Dashboard"):

    st.markdown("""

### Customer Churn Analytics Platform

Technology Stack

- Python
- Streamlit
- Plotly
- Pandas
- Scikit-learn
- XGBoost

Business KPIs

- Customer Churn
- Revenue at Risk
- Monthly Revenue
- Customer Lifetime
- Customer Segmentation

Machine Learning Models

- Logistic Regression

- Random Forest

- XGBoost

Dashboard Version

Enterprise Edition v1.0

""")