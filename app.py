import streamlit as st
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
