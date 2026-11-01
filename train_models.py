"""Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset (UCI),
save models to model/ and write test_data.csv + metrics table."""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42
data = load_breast_cancer(as_frame=True)
df = data.frame  # 30 features + 'target' (0 = malignant, 1 = benign)

X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Test data = features + target column (this is what gets uploaded to the app)
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False)),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.joblib")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
print(res.to_string(index=False))
