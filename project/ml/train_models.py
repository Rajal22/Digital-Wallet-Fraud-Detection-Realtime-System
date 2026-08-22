"""
Train the Isolation Forest + Autoencoder fraud detection models on synthetic
Nepal digital wallet transaction data, and save all artifacts needed by the
FastAPI backend.

This mirrors the original 01_fraud_detection notebook's approach.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

N_TRANSACTIONS = 50_000
N_USERS = 5_000

NEPAL_CITIES = [
    "Kathmandu", "Pokhara", "Lalitpur", "Bhaktapur", "Biratnagar",
    "Birgunj", "Dharan", "Butwal", "Nepalgunj", "Hetauda",
]
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "utility_bill", "mobile_topup", "ecommerce",
    "travel", "electronics", "fuel", "education", "healthcare",
]
DEVICE_TYPES = ["android", "ios", "web"]

print("Generating synthetic transaction data...")

user_ids = np.random.randint(1, N_USERS + 1, N_TRANSACTIONS)
account_age_days = np.random.randint(1, 1500, N_TRANSACTIONS)

df = pd.DataFrame({
    "transaction_id": [f"TXN{i:08d}" for i in range(N_TRANSACTIONS)],
    "user_id": user_ids,
    "amount": np.round(np.random.lognormal(mean=6.5, sigma=1.2, size=N_TRANSACTIONS), 2),
    "transaction_hour": np.random.randint(0, 24, N_TRANSACTIONS),
    "merchant_category": np.random.choice(MERCHANT_CATEGORIES, N_TRANSACTIONS),
    "device_type": np.random.choice(DEVICE_TYPES, N_TRANSACTIONS),
    "location": np.random.choice(NEPAL_CITIES, N_TRANSACTIONS),
    "is_new_device": np.random.choice([0, 1], N_TRANSACTIONS, p=[0.9, 0.1]),
    "failed_attempts": np.random.choice([0, 1, 2, 3, 4, 5], N_TRANSACTIONS,
                                         p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]),
    "transactions_last_hour": np.random.choice(range(1, 12), N_TRANSACTIONS,
                                                p=[0.4, 0.25, 0.15, 0.08, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005]),
    "account_age_days": account_age_days,
})

# --- Rule-based fraud labeling (evaluation only, not used for training) ---
def is_fraud(row):
    score = 0
    if row["amount"] > 15000:
        score += 1
    if row["transactions_last_hour"] >= 8:
        score += 1
    if row["is_new_device"] == 1 and row["amount"] > 10000:
        score += 1
    if row["failed_attempts"] >= 4:
        score += 1
    if row["account_age_days"] < 30 and row["amount"] > 8000:
        score += 1
    if row["amount"] > 12000 and row["transaction_hour"] in [0, 1, 2, 3]:
        score += 1
    return 1 if score >= 2 else (1 if (score == 1 and np.random.random() < 0.15) else 0)

df["is_fraud"] = df.apply(is_fraud, axis=1)
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# --- Feature engineering ---
df["is_night"] = df["transaction_hour"].isin([0, 1, 2, 3, 4]).astype(int)
df["amount_log"] = np.log1p(df["amount"])
df["high_velocity"] = (df["transactions_last_hour"] >= 8).astype(int)
df["high_failed_attempts"] = (df["failed_attempts"] >= 4).astype(int)
df["new_device_high_amount"] = ((df["is_new_device"] == 1) & (df["amount"] > 10000)).astype(int)
df["amount_vs_account_age"] = df["amount"] / (df["account_age_days"] + 1)
df["risk_factor_count"] = (
    df["is_night"] + df["high_velocity"] + df["high_failed_attempts"]
    + df["new_device_high_amount"] + (df["account_age_days"] < 30).astype(int)
)

FEATURE_COLUMNS = [
    "amount_log", "transaction_hour", "is_new_device", "failed_attempts",
    "transactions_last_hour", "account_age_days", "is_night",
    "high_velocity", "high_failed_attempts", "new_device_high_amount",
    "amount_vs_account_age", "risk_factor_count", "amount",
]

X = df[FEATURE_COLUMNS].values

# --- Scale features ---
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# --- Isolation Forest ---
print("Training Isolation Forest...")
contamination = float(df["is_fraud"].mean())
iso_forest = IsolationForest(
    n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1,
)
iso_forest.fit(X_scaled)

iso_raw_scores = -iso_forest.score_samples(X_scaled)  # higher = more anomalous
isolation_scaler = MinMaxScaler(feature_range=(0, 100))
iso_scores_normalized = isolation_scaler.fit_transform(iso_raw_scores.reshape(-1, 1)).flatten()

# --- Autoencoder (trained only on normal transactions) ---
print("Training Autoencoder...")
X_normal = X_scaled[df["is_fraud"] == 0]

n_features = X_scaled.shape[1]
inputs = keras.Input(shape=(n_features,))
encoded = layers.Dense(32, activation="relu")(inputs)
encoded = layers.Dense(16, activation="relu")(encoded)
encoded = layers.Dense(8, activation="relu")(encoded)
decoded = layers.Dense(16, activation="relu")(encoded)
decoded = layers.Dense(32, activation="relu")(decoded)
decoded = layers.Dense(n_features, activation="linear")(decoded)

autoencoder = keras.Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    X_normal, X_normal,
    epochs=30, batch_size=256, shuffle=True, verbose=0,
    validation_split=0.1,
)

reconstructions = autoencoder.predict(X_scaled, verbose=0)
reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
autoencoder_scaler = MinMaxScaler(feature_range=(0, 100))
ae_scores_normalized = autoencoder_scaler.fit_transform(reconstruction_errors.reshape(-1, 1)).flatten()

# --- Blended risk score ---
blended_score = 0.4 * iso_scores_normalized + 0.6 * ae_scores_normalized

normal_scores = blended_score[df["is_fraud"] == 0]
low_threshold = float(np.percentile(normal_scores, 95))
high_threshold = float(np.percentile(normal_scores, 99))

print(f"LOW threshold:  {low_threshold:.2f}")
print(f"HIGH threshold: {high_threshold:.2f}")

# --- Quick evaluation ---
def classify(score):
    if score < low_threshold:
        return "LOW"
    elif score < high_threshold:
        return "MEDIUM"
    return "HIGH"

risk_levels = pd.Series(blended_score).apply(classify)
print(pd.crosstab(risk_levels, df["is_fraud"], normalize="index"))

# --- Save all artifacts ---
OUT_DIR = "/home/claude/project/ml"
joblib.dump(iso_forest, f"{OUT_DIR}/isolation_forest.pkl")
autoencoder.save(f"{OUT_DIR}/fraud_autoencoder.keras")
joblib.dump(feature_scaler, f"{OUT_DIR}/feature_scaler.pkl")
joblib.dump(isolation_scaler, f"{OUT_DIR}/isolation_scaler.pkl")
joblib.dump(autoencoder_scaler, f"{OUT_DIR}/autoencoder_scaler.pkl")
joblib.dump(FEATURE_COLUMNS, f"{OUT_DIR}/feature_columns.pkl")
joblib.dump(
    {"low": low_threshold, "high": high_threshold, "iso_weight": 0.4, "ae_weight": 0.6},
    f"{OUT_DIR}/risk_config.pkl",
)

# Save a small sample of the synthetic data for the dashboard demo mode
df.sample(500, random_state=42).to_csv(f"{OUT_DIR}/sample_transactions.csv", index=False)

print("\nAll artifacts saved to", OUT_DIR)
import os
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)
