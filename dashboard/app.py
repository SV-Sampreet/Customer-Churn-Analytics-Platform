import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_churn_cleaned.csv"
)

df = pd.read_csv(csv_path)

# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Analytics Platform")

st.markdown(
    """
    Analyze customer churn behavior using
    machine learning, feature importance,
    and interactive visualizations.
    """
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🔍 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

customer_search = st.sidebar.text_input(
    "Search Customer ID"
)

filtered_df = df[
    df["Contract"].isin(contract_filter)
]

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 Business KPIs")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Customers",
    len(filtered_df)
)

churn_rate = (
    filtered_df["Churn"]
    .value_counts(normalize=True)
    .get("Yes", 0) * 100
)

col2.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col3.metric(
    "Avg Monthly Charges",
    f"${filtered_df['MonthlyCharges'].mean():.2f}"
)

# =====================================================
# CUSTOMER SEARCH
# =====================================================

if customer_search:

    result = filtered_df[
        filtered_df["customerID"].str.contains(
            customer_search,
            case=False,
            na=False
        )
    ]

    st.subheader("🔎 Customer Search Result")

    st.dataframe(result)

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df.head())

# =====================================================
# DATASET SHAPE
# =====================================================

st.subheader("📏 Dataset Shape")

st.write(filtered_df.shape)

# =====================================================
# CHURN DISTRIBUTION
# =====================================================

st.subheader("📉 Churn Distribution")

st.bar_chart(
    filtered_df["Churn"].value_counts()
)

# =====================================================
# PIE CHART
# =====================================================

st.subheader("🥧 Customer Churn Breakdown")

fig = px.pie(
    filtered_df,
    names="Churn",
    title="Customer Churn Breakdown"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# MONTHLY CHARGES DISTRIBUTION
# =====================================================

st.subheader("💰 Monthly Charges Distribution")

fig = px.histogram(
    filtered_df,
    x="MonthlyCharges",
    nbins=30,
    title="Monthly Charges Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TENURE DISTRIBUTION
# =====================================================

st.subheader("📅 Customer Tenure Distribution")

fig = px.histogram(
    filtered_df,
    x="tenure",
    nbins=30,
    title="Customer Tenure"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CHURN BY CONTRACT
# =====================================================

st.subheader("📄 Churn by Contract Type")

contract_churn = pd.crosstab(
    filtered_df["Contract"],
    filtered_df["Churn"]
)

st.bar_chart(contract_churn)

# =====================================================
# CHURN BY INTERNET SERVICE
# =====================================================

st.subheader("🌐 Churn by Internet Service")

internet_churn = pd.crosstab(
    filtered_df["InternetService"],
    filtered_df["Churn"]
)

st.bar_chart(internet_churn)

# =====================================================
# TOP CUSTOMERS
# =====================================================

st.subheader("🏆 Top 10 Customers by Monthly Charges")

top_customers = filtered_df.sort_values(
    by="MonthlyCharges",
    ascending=False
).head(10)

st.dataframe(
    top_customers[
        [
            "customerID",
            "MonthlyCharges",
            "Contract",
            "InternetService"
        ]
    ]
)

# =====================================================
# FEATURE IMPORTANCE IMAGE
# =====================================================

st.subheader("🔥 Top Churn Drivers")

feature_path = os.path.join(
    BASE_DIR,
    "images",
    "feature_importance.png"
)

if os.path.exists(feature_path):
    st.image(
        feature_path,
        use_container_width=True
    )
else:
    st.warning("feature_importance.png not found")

# =====================================================
# SHAP IMAGE
# =====================================================

st.subheader("🧠 SHAP Explainability")

shap_path = os.path.join(
    BASE_DIR,
    "images",
    "shap_summary.png"
)

if os.path.exists(shap_path):
    st.image(
        shap_path,
        use_container_width=True
    )
else:
    st.warning("shap_summary.png not found")

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.subheader("🤖 Model Performance")

performance = pd.DataFrame(
    {
        "Model": [
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            0.793,
            0.771
        ]
    }
)

st.dataframe(performance)

st.bar_chart(
    performance.set_index("Model")
)

# =====================================================
# DOWNLOAD BUTTON
# =====================================================

st.subheader("⬇️ Download Dataset")

csv_download = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv_download,
    file_name="telco_churn.csv",
    mime="text/csv"
)

# =====================================================
# PROJECT SUMMARY
# =====================================================

st.subheader("📌 Key Insights")

st.markdown("""
### Business Insights

- Customers with Fiber Optic Internet have higher churn risk.
- Month-to-Month contracts are strongly associated with churn.
- Long-tenure customers are more likely to stay.
- Electronic Check payment users churn more frequently.
- Contract duration is a major retention driver.

### Machine Learning Models

✅ Random Forest Classifier

✅ XGBoost Classifier

✅ SHAP Explainability

### Dataset

Telco Customer Churn Dataset
""")