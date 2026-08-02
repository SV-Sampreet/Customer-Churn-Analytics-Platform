import streamlit as st
import pandas as pd
import plotly.express as px
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
    page_title="Business Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"

)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/telco_churn_cleaned.csv")

df = load_data()

# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#059669,#047857);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:30px;
box-shadow:0 10px 30px rgba(5,150,105,.35);
">

<h1 style="margin:0;font-size:40px;">
📊 Business Insights Dashboard
</h1>

<p style="font-size:20px;margin-top:12px;">
Enterprise Business Intelligence & Customer Analytics
</p>

<p style="font-size:16px;color:#E5E7EB;margin-top:18px;">
Analyze customer behavior, churn trends, revenue performance, customer segmentation, and key business opportunities using interactive dashboards and visual analytics.
</p>

<hr style="border:1px solid rgba(255,255,255,.25);">

<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:25px;">

<div>
<b>📈 Analytics</b><br>
Customer Behaviour
</div>

<div>
<b>💰 Revenue</b><br>
Revenue Performance
</div>

<div>
<b>📉 Churn</b><br>
Retention Insights
</div>

<div>
<b>🎯 Business Goal</b><br>
Data-Driven Decisions
</div>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("🔍 Dashboard Filters")

gender = st.sidebar.multiselect(
    "Gender",
    sorted(df["gender"].unique()),
    default=sorted(df["gender"].unique())
)

contract = st.sidebar.multiselect(
    "Contract",
    sorted(df["Contract"].unique()),
    default=sorted(df["Contract"].unique())
)

internet = st.sidebar.multiselect(
    "Internet Service",
    sorted(df["InternetService"].unique()),
    default=sorted(df["InternetService"].unique())
)

payment = st.sidebar.multiselect(
    "Payment Method",
    sorted(df["PaymentMethod"].unique()),
    default=sorted(df["PaymentMethod"].unique())
)

churn = st.sidebar.multiselect(
    "Churn",
    sorted(df["Churn"].unique()),
    default=sorted(df["Churn"].unique())
)

# ==========================================================
# FILTER DATA
# ==========================================================

filtered = df[
    (df["gender"].isin(gender)) &
    (df["Contract"].isin(contract)) &
    (df["InternetService"].isin(internet)) &
    (df["PaymentMethod"].isin(payment)) &
    (df["Churn"].isin(churn))
]

# ==========================================================
# EXECUTIVE KPIs
# ==========================================================

customers = len(filtered)

churn_customers = (filtered["Churn"]=="Yes").sum()

active_customers = customers - churn_customers

churn_rate = churn_customers/customers*100 if customers>0 else 0

monthly_revenue = filtered["MonthlyCharges"].sum()

avg_monthly = filtered["MonthlyCharges"].mean()

avg_tenure = filtered["tenure"].mean()

revenue_risk = filtered.loc[
    filtered["Churn"]=="Yes",
    "MonthlyCharges"
].sum()

# ==========================================================
# KPI CARDS
# ==========================================================

r1 = st.columns(4)

r1[0].metric(
    "👥 Customers",
    f"{customers:,}"
)

r1[1].metric(
    "❌ Churn Customers",
    f"{churn_customers:,}"
)

r1[2].metric(
    "📉 Churn Rate",
    f"{churn_rate:.2f}%"
)

r1[3].metric(
    "💰 Monthly Revenue",
    f"${monthly_revenue:,.0f}"
)

r2 = st.columns(4)

r2[0].metric(
    "✅ Active Customers",
    f"{active_customers:,}"
)

r2[1].metric(
    "💵 Avg Monthly Charges",
    f"${avg_monthly:.2f}"
)

r2[2].metric(
    "📅 Avg Tenure",
    f"{avg_tenure:.1f} Months"
)

r2[3].metric(
    "⚠ Revenue At Risk",
    f"${revenue_risk:,.0f}"
)

st.divider()


# ==========================================================
# CUSTOMER 360 PROFILE
# ==========================================================

st.divider()

st.header("👤 Customer 360 Profile")

st.caption(
    "Explore an individual customer's profile, risk indicators and business value."
)

customer_index = st.selectbox(
    "Select Customer",
    filtered.index.tolist()
)

customer = filtered.loc[customer_index]

st.divider()

c1, c2 = st.columns([1, 2])

# ==========================================================
# CUSTOMER PROFILE
# ==========================================================

with c1:

    st.markdown("### 👤 Customer Profile")

    st.info(f"""
**Customer ID**

{customer_index}

---

**Gender**

{customer["gender"]}

---

**Senior Citizen**

{"Yes" if customer["SeniorCitizen"]==1 else "No"}

---

**Partner**

{customer["Partner"]}

---

**Dependents**

{customer["Dependents"]}
""")

# ==========================================================
# CUSTOMER DETAILS
# ==========================================================

with c2:

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Monthly Charges",
        f"${customer['MonthlyCharges']:.2f}"
    )

    k2.metric(
        "Tenure",
        f"{customer['tenure']} Months"
    )

    k3.metric(
        "Contract",
        customer["Contract"]
    )

    k4.metric(
        "Internet",
        customer["InternetService"]
    )

st.divider()

# ==========================================================
# BUSINESS SUMMARY
# ==========================================================

left, right = st.columns(2)

with left:

    st.success(f"""
### 💰 Customer Business Value

**Total Charges**

${customer["TotalCharges"]:.2f}

**Monthly Revenue**

${customer["MonthlyCharges"]:.2f}

**Customer Lifetime**

{customer["tenure"]} Months
""")

with right:

    risk = "🔴 High Risk" if customer["Churn"]=="Yes" else "🟢 Low Risk"

    st.warning(f"""
### 📋 Business Recommendation

**Current Status**

{risk}

### Recommended Actions

✅ Monitor customer engagement

✅ Offer annual contract

✅ Promote AutoPay

✅ Recommend premium services

✅ Schedule proactive follow-up
""")



# ==========================================================
# QUICK BUSINESS SUMMARY
# ==========================================================

left,right = st.columns([2,1])

with left:

    st.success(f"""
### 📈 Executive Summary

• Total Customers : **{customers:,}**

• Active Customers : **{active_customers:,}**

• Churn Customers : **{churn_customers:,}**

• Monthly Revenue : **${monthly_revenue:,.0f}**

• Revenue At Risk : **${revenue_risk:,.0f}**

• Average Customer Tenure : **{avg_tenure:.1f} Months**
""")

with right:

    st.info("""
### 💡 Business Goals

✅ Reduce Churn

✅ Increase Customer Lifetime Value

✅ Improve Retention

✅ Upsell Premium Plans

✅ Increase AutoPay Adoption
""")

st.divider()




# ==========================================================
# CHURN ANALYSIS
# ==========================================================

st.divider()

st.subheader("📊 Churn Analysis")

c1, c2 = st.columns(2)

with c1:

    contract_churn = (
        filtered.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        contract_churn,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        text_auto=True,
        title="Churn by Contract"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Contract Type",
        yaxis_title="Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:

    fig = px.pie(
        filtered,
        names="Churn",
        hole=0.55,
        title="Customer Churn Distribution",
        color="Churn",
        color_discrete_map={
            "Yes":"red",
            "No":"green"
        }
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# REVENUE ANALYSIS
# ==========================================================

st.subheader("💰 Revenue Analysis")

c1, c2 = st.columns(2)

with c1:

    revenue_contract = (
        filtered.groupby("Contract")["MonthlyCharges"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        revenue_contract,
        x="Contract",
        y="MonthlyCharges",
        color="MonthlyCharges",
        text_auto=".2s",
        title="Revenue by Contract"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

with c2:

    revenue_service = (
        filtered.groupby("InternetService")["MonthlyCharges"]
        .sum()
        .reset_index()
    )

    fig = px.sunburst(
        revenue_service,
        path=["InternetService"],
        values="MonthlyCharges",
        title="Revenue by Internet Service"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# CUSTOMER BEHAVIOUR
# ==========================================================

st.subheader("📈 Customer Behaviour")

c1, c2 = st.columns(2)

with c1:

    fig = px.scatter(
        filtered,
        x="MonthlyCharges",
        y="tenure",
        color="Churn",
        size="MonthlyCharges",
        hover_data=["Contract"],
        title="Monthly Charges vs Tenure"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

with c2:

    fig = px.box(
        filtered,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# SERVICE ANALYSIS
# ==========================================================

st.subheader("🌐 Service Analysis")

c1, c2 = st.columns(2)

with c1:

    fig = px.histogram(
        filtered,
        x="InternetService",
        color="Churn",
        barmode="group",
        title="Internet Service vs Churn"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

with c2:

    fig = px.histogram(
        filtered,
        x="PaymentMethod",
        color="Churn",
        barmode="group",
        title="Payment Method vs Churn"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# REVENUE TREEMAP
# ==========================================================

st.subheader("🌳 Revenue Treemap")

fig = px.treemap(
    filtered,
    path=["InternetService","Contract"],
    values="MonthlyCharges",
    color="MonthlyCharges",
    title="Revenue Distribution"
)

fig.update_layout(height=700)

st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

st.divider()

st.subheader("👥 Customer Segmentation")

seg1, seg2, seg3, seg4 = st.columns(4)

premium = filtered[filtered["MonthlyCharges"] >= 80]

loyal = filtered[filtered["tenure"] >= 48]

new_customers = filtered[filtered["tenure"] <= 12]

high_risk = filtered[
    (filtered["Contract"] == "Month-to-month") &
    (filtered["MonthlyCharges"] >= 80)
]

seg1.metric("💎 Premium", len(premium))
seg2.metric("❤️ Loyal", len(loyal))
seg3.metric("🆕 New", len(new_customers))
seg4.metric("⚠ High Risk", len(high_risk))

st.divider()

# ==========================================================
# TENURE DISTRIBUTION
# ==========================================================

st.subheader("📅 Customer Tenure Distribution")

fig = px.histogram(
    filtered,
    x="tenure",
    nbins=30,
    color="Churn",
    title="Customer Tenure Distribution"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# REVENUE AT RISK
# ==========================================================

st.subheader("💰 Revenue At Risk")

risk = (
    filtered[filtered["Churn"]=="Yes"]
    .groupby("Contract")["MonthlyCharges"]
    .sum()
    .reset_index()
)

fig = px.bar(
    risk,
    x="Contract",
    y="MonthlyCharges",
    color="MonthlyCharges",
    text_auto=".2s",
    title="Revenue At Risk by Contract"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# TOP HIGH RISK CUSTOMERS
# ==========================================================

st.subheader("🚨 Top High-Risk Customers")

risk_customers = filtered[
    (filtered["Contract"]=="Month-to-month") &
    (filtered["MonthlyCharges"]>=80)
].sort_values(
    by="MonthlyCharges",
    ascending=False
)

columns = [
    "gender",
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn"
]

st.dataframe(
    risk_customers[columns].head(10),
    use_container_width=True,
    height=350
)

st.divider()

# ==========================================================
# CUSTOMER LIFETIME VALUE
# ==========================================================

st.subheader("💎 Customer Lifetime Value (Estimated)")

clv = filtered.copy()

clv["CLV"] = clv["MonthlyCharges"] * clv["tenure"]

avg_clv = clv["CLV"].mean()

highest_clv = clv["CLV"].max()

c1, c2 = st.columns(2)

c1.metric(
    "Average CLV",
    f"${avg_clv:,.0f}"
)

c2.metric(
    "Highest CLV",
    f"${highest_clv:,.0f}"
)

fig = px.histogram(
    clv,
    x="CLV",
    nbins=40,
    title="Customer Lifetime Value Distribution"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# AI EXECUTIVE RECOMMENDATIONS
# ==========================================================

st.subheader("🤖 AI Executive Recommendations")

st.success(f"""
### 📈 Strategic Recommendations

✅ Churn Rate : **{churn_rate:.2f}%**

✅ Revenue At Risk : **${revenue_risk:,.0f}**

### Recommended Actions

• Focus retention campaigns on Month-to-Month customers.

• Convert Electronic Check customers to AutoPay.

• Reward customers with tenure greater than 48 months.

• Upsell Premium Internet plans to loyal customers.

• Provide discounts to high-value customers before renewal.

• Launch targeted offers for Fiber Optic users.

• Increase customer engagement during the first 12 months.

• Reduce churn through personalized retention campaigns.
""")

st.divider()

# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    filtered,
    use_container_width=True,
    height=400
)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.divider()

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered.select_dtypes(include=["number"])

if numeric_df.shape[1] > 1:

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix"
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Not enough numeric columns to display correlation heatmap.")



# ==========================================================
# REVENUE FORECAST (ESTIMATION)
# ==========================================================

st.divider()

st.subheader("📈 Revenue Forecast")

forecast = pd.DataFrame({
    "Month":[
        "Jan","Feb","Mar","Apr",
        "May","Jun","Jul","Aug",
        "Sep","Oct","Nov","Dec"
    ]
})

forecast["Revenue"] = [
    monthly_revenue * 0.95,
    monthly_revenue * 0.97,
    monthly_revenue * 1.00,
    monthly_revenue * 1.02,
    monthly_revenue * 1.04,
    monthly_revenue * 1.05,
    monthly_revenue * 1.07,
    monthly_revenue * 1.08,
    monthly_revenue * 1.10,
    monthly_revenue * 1.12,
    monthly_revenue * 1.13,
    monthly_revenue * 1.15
]

fig = px.line(
    forecast,
    x="Month",
    y="Revenue",
    markers=True,
    title="Estimated Revenue Forecast"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="payment_method_analysis"
)

# ==========================================================
# BUSINESS HEALTH SCORE
# ==========================================================

st.divider()

st.subheader("🏆 Business Health Score")

health = 100 - churn_rate

if health > 85:
    status = "Excellent"
elif health > 70:
    status = "Good"
elif health > 55:
    status = "Average"
else:
    status = "Needs Attention"

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health,
        title={"text":"Business Health"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"royalblue"},
            "steps":[
                {"range":[0,50],"color":"red"},
                {"range":[50,75],"color":"orange"},
                {"range":[75,100],"color":"green"}
            ]
        }
    )
)

gauge.update_layout(height=450)

left,right = st.columns([2,1])

with left:
    st.plotly_chart(
        gauge,
        use_container_width=True
    )

with right:

    st.metric(
        "Business Status",
        status
    )

    st.metric(
        "Health Score",
        f"{health:.1f}/100"
    )

# ==========================================================
# TOP PERFORMING SEGMENTS
# ==========================================================

st.divider()

st.subheader("🥇 Top Performing Segments")

segment = (
    filtered.groupby("Contract")
    .agg(
        Customers=("Contract","count"),
        Revenue=("MonthlyCharges","sum"),
        Avg_Tenure=("tenure","mean")
    )
    .reset_index()
)

st.dataframe(
    segment,
    use_container_width=True
)


# ==========================================================
# DOWNLOAD DATA
# ==========================================================

st.divider()

st.subheader("📥 Export Dashboard Data")

csv = filtered.to_csv(index=False)

st.download_button(
    "⬇ Download CSV Report",
    csv,
    "business_insights.csv",
    "text/csv",
    use_container_width=True
)

