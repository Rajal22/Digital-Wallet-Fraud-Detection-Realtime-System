# IEEE-CIS Fraud Detection (Supervised Classification)

A supervised fraud detection model trained on the real, labeled **IEEE-CIS Fraud Detection** dataset (Kaggle/Vesta Corporation) — using actual confirmed fraud cases rather than synthetic or rule-based labels.

## Overview

Unlike a cold-start scenario with no fraud history, this dataset comes with real, investigator-confirmed `isFraud` labels for ~590,000 e-commerce transactions. That changes the right approach entirely: instead of unsupervised anomaly detection, this project trains **supervised gradient-boosted tree models** (XGBoost and LightGBM) that learn directly from confirmed fraud examples.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — released by Vesta Corporation via Kaggle:

- **`train_transaction.csv`** — ~590,540 transactions × 394 columns: amount, card info, address, product code, email domains, and hundreds of anonymized engineered features (`C1-C14`, `D1-D15`, `V1-V339`)
- **`train_identity.csv`** — device/identity signals (device type, browser, OS) for a subset of transactions
- **`test_transaction.csv`** / **`test_identity.csv`** — the same structure, unlabeled, for scoring/submission
- **`sample_submission.csv`** — expected output format
- Real fraud rate: ~3.5%, heavily imbalanced

## Approach

1. **Memory-efficient loading** — the files are large (600+ MB) and wide (434 columns); loaded in chunks with immediate dtype downcasting to avoid `MemoryError` on typical machines, rather than loading the full file at default precision first
2. **Merge** `transaction` + `identity` tables on `TransactionID`
3. **Missing value handling** — columns with >90% missing values dropped; the rest left as `NaN`, since both XGBoost and LightGBM handle missing values natively and often use "is this missing" as a useful split
4. **Feature engineering**:
   - Time-of-day / day-of-week from `TransactionDT`
   - Log-transformed and decimal-component transaction amount
   - Email provider grouping (Gmail/Yahoo/Microsoft/etc.) and sender/recipient domain match
   - Frequency encoding for card/address identifiers
   - Label encoding for all categorical columns
5. **Time-based train/test split** (not random) — sorted by `TransactionDT`, holding out the most recent 20%, since a random split would leak future information into training (fraud data is a time series in disguise)
6. **Class imbalance handling** — `scale_pos_weight` in both models, rather than relying on plain accuracy
7. **Two independent models**: XGBoost and LightGBM, compared head-to-head on ROC-AUC and PR-AUC (PR-AUC matters more here given the class imbalance)
8. **Fault-tolerant training** — LightGBM training is wrapped so a broken/incompatible install (a real issue encountered during development, particularly on Windows) doesn't block the rest of the pipeline; it gracefully falls back to XGBoost-only
9. **Threshold tuning & business rules** — predicted probabilities converted into Approve / Verify / Block tiers using data-driven percentile thresholds, not an arbitrary 0.5 cutoff
10. **Test set scoring** — applies the exact same fitted preprocessing to the unlabeled test set, blends both models' predictions, and writes a Kaggle-format `submission.csv`

## Tech Stack

| Category | Tools |
|---|---|
| Data | NumPy, Pandas |
| Models | XGBoost, LightGBM |
| Evaluation | scikit-learn (ROC-AUC, PR-AUC, classification report, confusion matrix) |
| Visualization | Matplotlib, Seaborn |
| Persistence | joblib |

## Key Results

- Both models evaluated on ROC-AUC and PR-AUC on a genuine held-out time period, not a random split
- Feature importance analysis shows which signals (transaction amount, card frequency, time-of-day, etc.) each model relies on most
- Risk tiering (Approve/Verify/Block) built from the better-performing model's probability output, so a raw fraud probability becomes an actual operational decision

## Limitations

- Many features (`V1-V339`) are anonymized by the dataset provider — limits interpretability of exactly *why* a transaction is flagged, even though the model performs well
- Static risk thresholds; a real deployment would tune these against actual fraud-loss vs. false-positive-review costs
- Trained on 2017-2018 e-commerce transaction patterns — fraud patterns drift over time, so a production system would need periodic retraining

## Relationship to Other Projects

This is one of two fraud detection approaches in this body of work:

- **This project** — supervised learning on real, labeled data (IEEE-CIS). Shows the stronger approach available when real fraud labels exist.
- **[Digital-Wallet-Fraud-Detection-eSewa-Khalti-Context-](https://github.com/Rajal22/Digital-Wallet-Fraud-Detection-eSewa-Khalti-Context-)** and its production follow-up, **Digital-Wallet-Fraud-Detection-Realtime-System** — unsupervised anomaly detection (Isolation Forest + Autoencoder) on synthetic data, representing the realistic "cold start" case where no confirmed fraud labels exist yet for a new wallet provider.

Together, they demonstrate both ends of the fraud detection problem: what to do with no labels, and what to do once real ones exist.
