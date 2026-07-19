import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Credit Risk Benchmark Dataset.csv")

# ----------------------------
# Outlier Capping
# ----------------------------
def cap_outliers(series, cap=0.99):
    upper = series.quantile(cap)
    return np.where(series > upper, upper, series)

cols = [
    "rev_util",
    "debt_ratio",
    "late_30_59",
    "late_60_89",
    "late_90",
    "open_credit",
    "real_estate"
]

for col in cols:
    df[col] = cap_outliers(df[col])

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("dlq_2yrs", axis=1)
y = df["dlq_2yrs"]

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Save Model
# ----------------------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")