import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Load cleaned data
df = pd.read_csv("customer_behaviour.csv").drop_duplicates(subset=["customer_id"])
df["customer_age_group"] = df["customer_age_group"].fillna(df["customer_age_group"].mode()[0])
df["browsing_activity"] = df["browsing_activity"].fillna(df["browsing_activity"].mode()[0])
df["product_views"] = df["product_views"].fillna(df["product_views"].median())
df["session_duration"] = df["session_duration"].fillna(df["session_duration"].median())

cat_cols = ["customer_age_group", "browsing_activity", "customer_segment", "visitor_type", "month"]
num_cols = ["past_purchases", "product_views", "session_duration", "bounce_rate", "exit_rate", "page_values", "weekend"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

clf = LogisticRegression(max_iter=1000, random_state=42)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

X = df[cat_cols + num_cols]
y = df["purchase"].astype(int)

pipeline.fit(X, y)

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/fast_predictor.joblib")
print("Logistic Regression fast companion predictor saved successfully!")
