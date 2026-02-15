import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.write("App started...")

try:
    scaler = joblib.load("model/scaler.pkl")
    st.write("Scaler loaded")

    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

    st.write("Models loaded")

    st.title("Heart Disease Classification App")
    uploaded_file = st.file_uploader("Upload CSV (must contain target column)", type=["csv"])

    model_name = st.selectbox("Select Model", list(models.keys()))

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if "target" not in data.columns:
            st.error("Uploaded file must contain a 'target' column.")
            st.stop()

        st.write("CSV loaded")

        X = data.drop("target", axis=1)
        y = data["target"]

        X_scaled = scaler.transform(X)

        model = models[model_name]
        y_pred = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_scaled)

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.write("Accuracy:", acc)
        st.write("AUC:", auc)
        st.write("Precision:", prec)
        st.write("Recall:", rec)
        st.write("F1:", f1)
        st.write("MCC:", mcc)

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
