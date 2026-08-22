import numpy as np
import joblib
import tensorflow as tf
from . import config


class FraudScorer:
    """Loads the trained Isolation Forest + Autoencoder and scores transactions
    in real time using the exact same feature engineering as training."""

    def __init__(self, ml_dir: str = config.ML_DIR):
        self.iso_forest = joblib.load(f"{ml_dir}/isolation_forest.pkl")
        self.autoencoder = tf.keras.models.load_model(f"{ml_dir}/fraud_autoencoder.keras")
        self.feature_scaler = joblib.load(f"{ml_dir}/feature_scaler.pkl")
        self.isolation_scaler = joblib.load(f"{ml_dir}/isolation_scaler.pkl")
        self.autoencoder_scaler = joblib.load(f"{ml_dir}/autoencoder_scaler.pkl")
        self.feature_columns = joblib.load(f"{ml_dir}/feature_columns.pkl")
        risk_config = joblib.load(f"{ml_dir}/risk_config.pkl")
        self.low_threshold = risk_config["low"]
        self.high_threshold = risk_config["high"]
        self.iso_weight = risk_config["iso_weight"]
        self.ae_weight = risk_config["ae_weight"]

    def _engineer_features(self, txn: dict) -> np.ndarray:
        amount = txn["amount"]
        hour = txn["transaction_hour"]
        is_new_device = txn["is_new_device"]
        failed_attempts = txn["failed_attempts"]
        txns_last_hour = txn["transactions_last_hour"]
        account_age_days = txn["account_age_days"]

        is_night = 1 if hour in (0, 1, 2, 3, 4) else 0
        amount_log = np.log1p(amount)
        high_velocity = 1 if txns_last_hour >= 8 else 0
        high_failed_attempts = 1 if failed_attempts >= 4 else 0
        new_device_high_amount = 1 if (is_new_device == 1 and amount > 10000) else 0
        amount_vs_account_age = amount / (account_age_days + 1)
        risk_factor_count = (
            is_night + high_velocity + high_failed_attempts
            + new_device_high_amount + (1 if account_age_days < 30 else 0)
        )

        feature_map = {
            "amount_log": amount_log,
            "transaction_hour": hour,
            "is_new_device": is_new_device,
            "failed_attempts": failed_attempts,
            "transactions_last_hour": txns_last_hour,
            "account_age_days": account_age_days,
            "is_night": is_night,
            "high_velocity": high_velocity,
            "high_failed_attempts": high_failed_attempts,
            "new_device_high_amount": new_device_high_amount,
            "amount_vs_account_age": amount_vs_account_age,
            "risk_factor_count": risk_factor_count,
            "amount": amount,
        }
        return np.array([[feature_map[c] for c in self.feature_columns]])

    def score(self, txn: dict) -> dict:
        X = self._engineer_features(txn)
        X_scaled = self.feature_scaler.transform(X)

        iso_raw = -self.iso_forest.score_samples(X_scaled)
        iso_score = float(np.clip(self.isolation_scaler.transform(iso_raw.reshape(-1, 1))[0, 0], 0, 100))

        reconstruction = self.autoencoder.predict(X_scaled, verbose=0)
        recon_error = np.mean(np.square(X_scaled - reconstruction), axis=1)
        ae_score = float(np.clip(self.autoencoder_scaler.transform(recon_error.reshape(-1, 1))[0, 0], 0, 100))

        blended = self.iso_weight * iso_score + self.ae_weight * ae_score

        if blended < self.low_threshold:
            risk_level, action = "LOW", config.LOW_ACTION
        elif blended < self.high_threshold:
            risk_level, action = "MEDIUM", config.MEDIUM_ACTION
        else:
            risk_level, action = "HIGH", config.HIGH_ACTION

        return {
            "risk_score": round(blended, 2),
            "isolation_score": round(iso_score, 2),
            "autoencoder_score": round(ae_score, 2),
            "risk_level": risk_level,
            "action": action,
        }


# Singleton, loaded once at app startup
scorer: FraudScorer | None = None


def get_scorer() -> FraudScorer:
    global scorer
    if scorer is None:
        scorer = FraudScorer()
    return scorer
