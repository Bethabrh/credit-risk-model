import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# -----------------------------
# 1. FEATURE ENGINEERING
# -----------------------------
def create_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    customer_features = df.groupby("CustomerId").agg(
        TotalTransactionAmount=("Amount", "sum"),
        AverageTransactionAmount=("Amount", "mean"),
        TransactionCount=("TransactionId", "count"),
        StdTransactionAmount=("Amount", "std")
    ).reset_index()

    df = df.merge(customer_features, on="CustomerId", how="left")

    df["StdTransactionAmount"] = df["StdTransactionAmount"].fillna(0)

    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    return df


# -----------------------------
# 2. PIPELINE BUILDER
# -----------------------------
def build_pipeline():

    num_features = [
        "Amount", "Value",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear"
    ]

    cat_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])


# -----------------------------
# 3. FINAL FUNCTION
# -----------------------------
def get_processed_data(df):

    df = create_features(df)

    X = df.drop(columns=["FraudResult"], errors="ignore")

    pipeline = build_pipeline()

    X_processed = pipeline.fit_transform(X)

    return X_processed, df