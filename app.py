import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report)

st.set_page_config(page_title="ML Classification Demo", layout="wide")
st.title("Breast Cancer Classification – 5 ML Models")
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic) – UCI | Target: 0 = Malignant, 1 = Benign")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "target"


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:  # only one class present in uploaded data
        auc = np.nan
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


# ---------- Sidebar ----------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")

if uploaded is None:
    st.info("Upload the test CSV (features + `target` column) from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"The CSV must contain the target column named '{TARGET}'.")
    st.stop()

X, y = df.drop(columns=TARGET), df[TARGET]
st.subheader("Uploaded data preview")
st.write(f"Rows: {df.shape[0]} | Features: {X.shape[1]}")
st.dataframe(df.head(), width="stretch")

# ---------- Selected model ----------
model = load_model(model_name)
try:
    X = X[list(model.feature_names_in_)]  # align column order
    pred, metrics = compute_metrics(model, X, y)
except Exception as e:
    st.error(f"Could not score the data: {e}")
    st.stop()

st.subheader(f"Evaluation metrics – {model_name}")
cols = st.columns(6)
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, "N/A" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=["Malignant", "Benign"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

# ---------- All models comparison ----------
st.subheader("Comparison of all 5 models on the uploaded data")
rows = []
for name in MODEL_FILES:
    _, m = compute_metrics(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
st.dataframe(pd.DataFrame(rows).set_index("ML Model Name").round(4), width="stretch")
