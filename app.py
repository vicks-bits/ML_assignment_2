"""import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.title("Adult Income Classification")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic_Regression",
     "Decision_Tree",
     "KNN",
     "Naive_Bayes",
     "Random_Forest",
     "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    model = joblib.load(f"model/{model_name}.pkl")

    X = data.drop("income", axis=1)
    y = data["income"]

    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    st.pyplot(fig)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification - ML Assignment 2")

# =====================================================
# Load Training Data (ONLY for preprocessing consistency)
# =====================================================
@st.cache_data
def load_training_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race",
        "sex","capital-gain","capital-loss","hours-per-week",
        "native-country","income"
    ]

    df = pd.read_csv(url, header=None, names=columns, na_values=" ?")
    df.dropna(inplace=True)
    return df


df_train = load_training_data()

# =====================================================
# Fit LabelEncoders & Scaler based on training data
# =====================================================
label_encoders = {}

for col in df_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    label_encoders[col] = le

X_train = df_train.drop("income", axis=1)
scaler = StandardScaler()
scaler.fit(X_train)


# =====================================================
# Download Sample Test Dataset Button
# =====================================================
@st.cache_data
def load_sample_test():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race",
        "sex","capital-gain","capital-loss","hours-per-week",
        "native-country","income"
    ]

    df_test = pd.read_csv(url, header=0, names=columns, na_values=" ?")
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
    df_test.dropna(inplace=True)

    return df_test


sample_test_df = load_sample_test()

st.download_button(
    label="Download Sample Test Dataset",
    data=sample_test_df.to_csv(index=False),
    file_name="adult_sample_test.csv",
    mime="text/csv"
)


# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# =====================================================
# Load Selected Model
# =====================================================
if uploaded_file is not None:

    model = joblib.load(f"model/{model_name}.pkl")

    df_test = pd.read_csv(uploaded_file)

    # =============================
    # Apply same preprocessing
    # =============================
    for col in df_test.select_dtypes(include='object').columns:
        if col in label_encoders:
            df_test[col] = label_encoders[col].transform(df_test[col])

    X_test = df_test.drop("income", axis=1)
    y_test = df_test["income"]

    X_test_scaled = scaler.transform(X_test)

    # =============================
    # Predictions
    # =============================
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # =============================
    # Metrics
    # =============================
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # =============================
    # Confusion Matrix
    # =============================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

    # =============================
    # Classification Report
    # =============================
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
