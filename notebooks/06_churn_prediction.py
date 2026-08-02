#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import joblib

plt.style.use("ggplot")

print("Libraries imported successfully!")


# In[2]:


df = pd.read_csv("../data/processed/telco_churn_feature_engineered.csv")

print("Dataset loaded successfully!")


# In[3]:


df.head()


# In[4]:


print(df.shape)

df.info()


# In[5]:


X = df.drop("Churn", axis=1)

y = df["Churn"]

print(X.shape)

print(y.shape)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Set :", X_train.shape)

print("Testing Set :", X_test.shape)


# In[7]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling completed!")


# In[8]:


print("="*50)

print("Train Class Distribution")

print(y_train.value_counts())

print("="*50)


# In[9]:


print("="*50)

print("Test Class Distribution")

print(y_test.value_counts())

print("="*50)


# In[10]:


logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression trained successfully!")


# In[11]:


logistic_predictions = logistic_model.predict(
    X_test_scaled
)

logistic_probabilities = logistic_model.predict_proba(
    X_test_scaled
)[:,1]


# In[12]:


print("="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)

print("Accuracy :", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions))
print("Recall   :", recall_score(y_test, logistic_predictions))
print("F1 Score :", f1_score(y_test, logistic_predictions))
print("ROC AUC  :", roc_auc_score(y_test, logistic_probabilities))


# In[13]:


print(classification_report(
    y_test,
    logistic_predictions
))


# In[14]:


cm = confusion_matrix(
    y_test,
    logistic_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[15]:


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully!")


# In[16]:


rf_predictions = rf_model.predict(
    X_test
)

rf_probabilities = rf_model.predict_proba(
    X_test
)[:,1]


# In[17]:


print("="*60)
print("RANDOM FOREST RESULTS")
print("="*60)

print("Accuracy :", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("Recall   :", recall_score(y_test, rf_predictions))
print("F1 Score :", f1_score(y_test, rf_predictions))
print("ROC AUC  :", roc_auc_score(y_test, rf_probabilities))


# In[18]:


print(classification_report(
    y_test,
    rf_predictions
))


# In[19]:


cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[20]:


xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost trained successfully!")


# In[21]:


xgb_predictions = xgb_model.predict(
    X_test
)

xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:,1]


# In[22]:


print("="*60)
print("XGBOOST RESULTS")
print("="*60)

print("Accuracy :", accuracy_score(y_test, xgb_predictions))
print("Precision:", precision_score(y_test, xgb_predictions))
print("Recall   :", recall_score(y_test, xgb_predictions))
print("F1 Score :", f1_score(y_test, xgb_predictions))
print("ROC AUC  :", roc_auc_score(y_test, xgb_probabilities))


# In[23]:


print(classification_report(
    y_test,
    xgb_predictions
))


# In[24]:


cm = confusion_matrix(
    y_test,
    xgb_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("XGBoost Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[25]:


models = {
    "Logistic Regression": (
        logistic_model,
        X_train_scaled
    ),
    "Random Forest": (
        rf_model,
        X_train
    ),
    "XGBoost": (
        xgb_model,
        X_train
    )
}

cv_results = []

for model_name, (model, X_data) in models.items():

    scores = cross_val_score(
        model,
        X_data,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Mean CV Accuracy": scores.mean(),
        "Std": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df


# In[26]:


param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [8, 10, 12],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[27]:


best_rf = grid_search.best_estimator_

best_rf_predictions = best_rf.predict(X_test)

best_rf_probabilities = best_rf.predict_proba(X_test)[:,1]

print(classification_report(
    y_test,
    best_rf_predictions
))


# In[28]:


leaderboard = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy":[
        accuracy_score(y_test, logistic_predictions),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test, xgb_predictions)
    ],

    "Precision":[
        precision_score(y_test, logistic_predictions),
        precision_score(y_test, rf_predictions),
        precision_score(y_test, xgb_predictions)
    ],

    "Recall":[
        recall_score(y_test, logistic_predictions),
        recall_score(y_test, rf_predictions),
        recall_score(y_test, xgb_predictions)
    ],

    "F1 Score":[
        f1_score(y_test, logistic_predictions),
        f1_score(y_test, rf_predictions),
        f1_score(y_test, xgb_predictions)
    ],

    "ROC AUC":[
        roc_auc_score(y_test, logistic_probabilities),
        roc_auc_score(y_test, rf_probabilities),
        roc_auc_score(y_test, xgb_probabilities)
    ]

})

leaderboard = leaderboard.sort_values(
    "ROC AUC",
    ascending=False
)

leaderboard


# In[29]:


leaderboard.to_csv(
    "../reports/model_comparison.csv",
    index=False
)

print("Metrics saved successfully!")


# In[30]:


joblib.dump(
    logistic_model,
    "../models/logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    "../models/random_forest.pkl"
)

joblib.dump(
    xgb_model,
    "../models/xgboost.pkl"
)

print("Models saved successfully!")


# In[31]:


plt.figure(figsize=(10,7))

fpr, tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

fpr, tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

plt.plot(
    fpr,
    tpr,
    label="Random Forest"
)

fpr, tpr, _ = roc_curve(
    y_test,
    xgb_probabilities
)

plt.plot(
    fpr,
    tpr,
    label="XGBoost"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()


# In[32]:


plt.figure(figsize=(10,7))

precision, recall, _ = precision_recall_curve(
    y_test,
    logistic_probabilities
)

plt.plot(
    recall,
    precision,
    label="Logistic Regression"
)

precision, recall, _ = precision_recall_curve(
    y_test,
    rf_probabilities
)

plt.plot(
    recall,
    precision,
    label="Random Forest"
)

precision, recall, _ = precision_recall_curve(
    y_test,
    xgb_probabilities
)

plt.plot(
    recall,
    precision,
    label="XGBoost"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.show()


# In[33]:


best_model = leaderboard.iloc[0]

print("="*60)
print("BEST MODEL")
print("="*60)

print(best_model)


# In[34]:


print("="*60)
print("MODEL TRAINING COMPLETED")
print("="*60)

print("Total Models Trained :", len(leaderboard))
print("Best Model :", leaderboard.iloc[0]["Model"])
print("Best ROC AUC :", round(leaderboard.iloc[0]["ROC AUC"],4))


# In[35]:


importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":rf_model.feature_importances_

})

importance = importance.sort_values(

    "Importance",

    ascending=False

)

importance.head(20)


# In[36]:


plt.figure(figsize=(12,7))

sns.barplot(

    data=importance.head(15),

    x="Importance",

    y="Feature"

)

plt.title("Random Forest Feature Importance")

plt.show()


# In[37]:


import time

start = time.time()

rf_model.fit(

    X_train,

    y_train

)

end = time.time()

print("Training Time :",end-start,"seconds")


# In[38]:


start = time.time()

rf_model.predict(

    X_test

)

end = time.time()

print("Prediction Time :",end-start,"seconds")


# In[39]:


importance.to_csv(

    "../reports/feature_importance.csv",

    index=False

)

print("Feature importance saved.")


# In[40]:


import joblib

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("Scaler saved successfully!")


# In[ ]:




