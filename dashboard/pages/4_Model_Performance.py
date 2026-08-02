import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


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
    page_title="Model Performance",
    page_icon="📈",
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
y = df["Churn"]

# ==========================================================
# LOAD MODELS
# ==========================================================

xgb = joblib.load("models/xgboost.pkl")
rf = joblib.load("models/random_forest.pkl")
lr = joblib.load("models/logistic_regression.pkl")

# ==========================================================
# PREDICTIONS
# ==========================================================

xgb_pred = xgb.predict(X)
xgb_prob = xgb.predict_proba(X)[:,1]

rf_pred = rf.predict(X)
rf_prob = rf.predict_proba(X)[:,1]

lr_pred = lr.predict(X)
lr_prob = lr.predict_proba(X)[:,1]

# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#DC2626,#B91C1C);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:30px;
box-shadow:0 10px 30px rgba(220,38,38,.35);
">

<h1 style="margin:0;font-size:40px;">
📈 Model Performance Dashboard
</h1>

<p style="font-size:20px;margin-top:12px;">
Machine Learning Model Evaluation & Performance Analytics
</p>

<p style="font-size:16px;color:#E5E7EB;margin-top:18px;">
Compare multiple Machine Learning models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, ROC Curve, Precision-Recall Curve, and Feature Importance.
</p>

<hr style="border:1px solid rgba(255,255,255,.25);">

<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:25px;">

<div>
<b>🤖 Models</b><br>
Logistic Regression<br>
Random Forest<br>
XGBoost
</div>

<div>
<b>📊 Metrics</b><br>
Accuracy<br>
Precision<br>
Recall
</div>

<div>
<b>📈 Evaluation</b><br>
ROC Curve<br>
PR Curve<br>
Confusion Matrix
</div>

<div>
<b>🏆 Goal</b><br>
Select the Best Performing Model
</div>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# MODEL METRICS
# ==========================================================

def metrics(name, y_true, pred, prob):

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1 Score": f1_score(y_true, pred),
        "ROC AUC": roc_auc_score(y_true, prob)
    }

results = pd.DataFrame([
    metrics("Logistic Regression", y, lr_pred, lr_prob),
    metrics("Random Forest", y, rf_pred, rf_prob),
    metrics("XGBoost", y, xgb_pred, xgb_prob)
])

# ==========================================================
# BUSINESS IMPACT CALCULATIONS
# ==========================================================

total_customers = len(y)

avg_monthly_revenue = 65

current_annual_revenue = (
    total_customers *
    avg_monthly_revenue *
    12
)

current_churn = (
    (y == "Yes").sum() /
    total_customers
)

revenue_lost = (
    current_annual_revenue *
    current_churn
)

expected_reduction = 0.10

revenue_protected = (
    revenue_lost *
    expected_reduction
)

projected_revenue = (
    current_annual_revenue -
    revenue_lost +
    revenue_protected
)


# ==========================================================
# KPI CARDS
# ==========================================================

best = results.sort_values(
    "ROC AUC",
    ascending=False
).iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🏆 Best Model",
    best["Model"]
)

c2.metric(
    "🎯 Accuracy",
    f"{best['Accuracy']*100:.2f}%"
)

c3.metric(
    "📈 ROC AUC",
    f"{best['ROC AUC']:.3f}"
)

c4.metric(
    "⭐ F1 Score",
    f"{best['F1 Score']:.3f}"
)

st.divider()


st.header("📈 Revenue Waterfall & KPI Forecast")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Annual Revenue",
    f"${current_annual_revenue:,.0f}"
)

k2.metric(
    "Revenue Lost",
    f"${revenue_lost:,.0f}"
)

k3.metric(
    "Revenue Protected",
    f"${revenue_protected:,.0f}"
)

k4.metric(
    "Projected Revenue",
    f"${projected_revenue:,.0f}"
)


# ==========================================================
# WATERFALL CHART
# ==========================================================

fig = go.Figure(go.Waterfall(

    name="Revenue",

    orientation="v",

    measure=[
        "absolute",
        "relative",
        "relative",
        "total"
    ],

    x=[
        "Current Revenue",
        "Revenue Lost",
        "Revenue Protected",
        "Projected Revenue"
    ],

    y=[
        current_annual_revenue,
        -revenue_lost,
        revenue_protected,
        projected_revenue
    ],

    connector={
        "line": {
            "color": "gray"
        }
    }

))

fig.update_layout(
    height=500,
    title="Revenue Impact Simulation"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# EXECUTIVE FORECAST
# ==========================================================

st.success(f"""
## 📋 Executive Revenue Forecast

### Current Annual Revenue

**${current_annual_revenue:,.0f}**

---

### Revenue Lost Due To Churn

**${revenue_lost:,.0f}**

---

### Revenue Protected

**${revenue_protected:,.0f}**

---

### Projected Annual Revenue

**${projected_revenue:,.0f}**

---

### Strategic Recommendations

✅ Focus retention campaigns on high-risk customers.

✅ Increase AutoPay adoption.

✅ Promote One-Year and Two-Year contracts.

✅ Continuously monitor churn trends.

✅ Retrain the model with new customer data.
""")


# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================

st.subheader("📋 Model Comparison")

display = results.copy()

for col in display.columns[1:]:
    display[col] = display[col].round(4)

st.dataframe(
    display,
    use_container_width=True,
    height=220
)

st.divider()

# ==========================================================
# MODEL COMPARISON CHARTS
# ==========================================================

st.subheader("📊 Model Performance Comparison")

fig = px.bar(
    results.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    ),
    x="variable",
    y="value",
    color="Model",
    barmode="group",
    text_auto=".3f",
    title="Comparison of ML Models"
)

fig.update_layout(
    height=550,
    xaxis_title="Evaluation Metric",
    yaxis_title="Score",
    yaxis=dict(range=[0,1.05])
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# ACCURACY COMPARISON
# ==========================================================

st.subheader("🎯 Accuracy Comparison")

accuracy_df = results[["Model","Accuracy"]]

fig = px.bar(
    accuracy_df,
    x="Model",
    y="Accuracy",
    color="Accuracy",
    text_auto=".3f",
    title="Accuracy by Model"
)

fig.update_layout(
    height=450,
    yaxis=dict(range=[0,1.05])
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# ROC AUC COMPARISON
# ==========================================================

st.subheader("📈 ROC-AUC Comparison")

fig = px.bar(
    results,
    x="Model",
    y="ROC AUC",
    color="ROC AUC",
    text_auto=".3f",
    title="ROC-AUC Score"
)

fig.update_layout(
    height=450,
    yaxis=dict(range=[0,1.05])
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# MODEL RANKING
# ==========================================================

st.subheader("🏆 Model Ranking")

ranking = results.sort_values(
    "ROC AUC",
    ascending=False
).reset_index(drop=True)

ranking.index += 1

st.dataframe(
    ranking,
    use_container_width=True,
    height=220
)

winner = ranking.iloc[0]

st.success(f"""
### 🥇 Best Performing Model

**{winner['Model']}**

Accuracy : **{winner['Accuracy']:.3f}**

Precision : **{winner['Precision']:.3f}**

Recall : **{winner['Recall']:.3f}**

F1 Score : **{winner['F1 Score']:.3f}**

ROC AUC : **{winner['ROC AUC']:.3f}**
""")

st.divider()

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y, xgb_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No","Actual Yes"],
    columns=["Predicted No","Predicted Yes"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    title="XGBoost Confusion Matrix"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================================
# ROC CURVE
# ==========================================================

st.subheader("📈 ROC Curve")

fpr_lr,tpr_lr,_ = roc_curve(y,lr_prob)
fpr_rf,tpr_rf,_ = roc_curve(y,rf_prob)
fpr_xgb,tpr_xgb,_ = roc_curve(y,xgb_prob)

roc = go.Figure()

roc.add_trace(
    go.Scatter(
        x=fpr_lr,
        y=tpr_lr,
        mode="lines",
        name="Logistic Regression"
    )
)

roc.add_trace(
    go.Scatter(
        x=fpr_rf,
        y=tpr_rf,
        mode="lines",
        name="Random Forest"
    )
)

roc.add_trace(
    go.Scatter(
        x=fpr_xgb,
        y=tpr_xgb,
        mode="lines",
        name="XGBoost"
    )
)

roc.add_trace(
    go.Scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        name="Random",
        line=dict(dash="dash")
    )
)

roc.update_layout(
    title="ROC Curve Comparison",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=600
)

st.plotly_chart(
    roc,
    use_container_width=True
)

st.divider()

# ==========================================================
# PRECISION RECALL CURVE
# ==========================================================

st.subheader("📉 Precision Recall Curve")

p1,r1,_ = precision_recall_curve(y,lr_prob)
p2,r2,_ = precision_recall_curve(y,rf_prob)
p3,r3,_ = precision_recall_curve(y,xgb_prob)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=r1,
        y=p1,
        mode="lines",
        name="Logistic Regression"
    )
)

fig.add_trace(
    go.Scatter(
        x=r2,
        y=p2,
        mode="lines",
        name="Random Forest"
    )
)

fig.add_trace(
    go.Scatter(
        x=r3,
        y=p3,
        mode="lines",
        name="XGBoost"
    )
)

fig.update_layout(
    title="Precision Recall Curve",
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.subheader("⭐ Top 20 Important Features")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":xgb.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).head(20)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    text_auto=".3f",
    title="Top 20 Features"
)

fig.update_layout(height=700)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.subheader("🤖 Executive Summary")

best = results.sort_values(
    "ROC AUC",
    ascending=False
).iloc[0]

st.success(f"""
### 🏆 Best Model : {best['Model']}

**Accuracy:** {best['Accuracy']:.3f}

**Precision:** {best['Precision']:.3f}

**Recall:** {best['Recall']:.3f}

**F1 Score:** {best['F1 Score']:.3f}

**ROC AUC:** {best['ROC AUC']:.3f}

---

### Business Recommendation

✅ Deploy **XGBoost** for production predictions.

✅ Continue monitoring model performance.

✅ Retrain monthly with new customer data.

✅ Focus retention efforts on high-risk customers predicted by the model.

✅ Use SHAP Explainability to interpret predictions.
""")