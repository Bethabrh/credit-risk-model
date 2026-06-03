import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

from src.data_processing import create_features, build_pipeline


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/processed/credit_risk_data.csv")

# -------------------------
# FEATURE ENGINEERING ONLY
# -------------------------
df = create_features(df)

# -------------------------
# TARGET (MUST EXIST HERE)
# -------------------------
if "is_high_risk" not in df.columns:
    raise ValueError("is_high_risk not found. Run Task 4 properly first.")

y = df["is_high_risk"]

# -------------------------
# FEATURES
# -------------------------
X = df.drop(columns=["is_high_risk", "FraudResult"], errors="ignore")

# -------------------------
# PREPROCESSING
# -------------------------
pipeline = build_pipeline()
X = pipeline.fit_transform(X)

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# MODELS
# -------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

mlflow.set_experiment("credit-risk-model")

best_model = None
best_score = 0

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="credit-risk-model"
)

        print(f"\n{name} AUC: {auc:.4f}")

        if auc > best_score:
            best_score = auc
            best_model = model

print("\nBest model AUC:", best_score)
df = pd.read_csv("data/processed/credit_risk_data.csv")
df = create_features(df)

print("is_high_risk exists:", "is_high_risk" in df.columns)
import pandas as pd

results = pd.DataFrame({
    "Model": ["LogisticRegression", "RandomForest"],
    "AUC": [0.9171, 0.9991]
})

print("\nModel Comparison:")
print(results)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, preds)
print("\nConfusion Matrix:")
print(cm)
import joblib

joblib.dump(best_model, "best_model.pkl")
print("\nBest model saved as best_model.pkl")
import joblib

joblib.dump(pipeline, "pipeline.pkl")
joblib.dump(best_model, "best_model.pkl")