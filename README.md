# Real-Time Digital Wallet Fraud Detection System

A production-shaped fraud detection system for Nepal digital wallets (eSewa/Khalti-style): unsupervised ML scoring (Isolation Forest + Autoencoder), a FastAPI backend for real-time transaction flagging/blocking, Cassandra (via DataStax Astra DB) for storage, and a Streamlit live monitoring dashboard.

| | |
|---|---|
| **Difficulty** | Advanced |
| **Industry** | Finance / Tech |
| **Target users** | Payment gateways, banks, end users |
| **ML** | Isolation Forest (scikit-learn) |
| **Deep Learning** | Autoencoder (TensorFlow/Keras) |
| **Backend** | FastAPI |
| **Database** | Cassandra via DataStax Astra DB (Data API) |
| **Frontend** | Streamlit (real-time dashboard) |
| **Deployment** | Google Cloud Platform |

## Architecture

```
┌────────────────┐      POST /transactions/score       ┌──────────────────┐
│   Streamlit     │ ──────────────────────────────────▶ │  FastAPI Backend │
│   Dashboard     │ ◀────────────────────────────────── │  (Isolation      │
│  (live monitor) │      GET /stats, /transactions/recent│  Forest +        │
└─────────────────┘                                      │  Autoencoder)    │
                                                           └────────┬─────────┘
                                                                    │ insert / query
                                                                    ▼
                                                           ┌──────────────────┐
                                                           │  Astra DB        │
                                                           │  (Cassandra,     │
                                                           │  Data API)       │
                                                           └──────────────────┘
```

**Why this shape:**
- **Isolation Forest + Autoencoder** — no real fraud labels exist yet for a new wallet (cold start), so unsupervised anomaly detection is the right tool. Fraud "labels" only exist as rule-based heuristics used for evaluation, never for training.
- **FastAPI** — a stateless scoring service that loads both models once at startup and scores each incoming transaction in milliseconds.
- **Astra DB (Cassandra)** — write-heavy, append-only transaction logging is exactly Cassandra's strength. Astra's modern Data API (via the `astrapy` library) is used instead of raw CQL — no secure connect bundle, no C-extension driver to compile, which sidesteps a whole class of install problems on Windows.
- **Streamlit** — fast to build a real-time ops dashboard without a separate frontend framework; polls the API and renders live charts/feed.

## Project Structure

```
fraud-detection-system/
├── ml/
│   ├── train_models.py        # generates synthetic data, trains & saves all model artifacts
│   ├── isolation_forest.pkl
│   ├── fraud_autoencoder.keras
│   ├── feature_scaler.pkl
│   ├── isolation_scaler.pkl
│   ├── autoencoder_scaler.pkl
│   ├── feature_columns.pkl
│   └── risk_config.pkl
├── notebooks/
│   └── ieee_fraud_detection.ipynb   # separate exploration: supervised fraud
│                                     # classification on the real IEEE-CIS dataset
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI routes
│   │   ├── ml_models.py       # loads models, scores transactions in real time
│   │   ├── database.py        # in-memory / local Cassandra / Astra DB storage
│   │   ├── models.py          # Pydantic request/response schemas
│   │   └── config.py          # loads settings, including .env via python-dotenv
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── dashboard.py            # Streamlit live monitoring dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml           # local dev: Cassandra + backend + frontend together
├── .env                         # your real secrets — NEVER commit this
├── .env.example                 # template showing what .env needs, safe to commit
└── README.md
```

## Quickstart (local, zero setup)

The fastest way to see it working — no Cassandra, no Docker, just Python:

```bash
# 1. Train and save the models (run once)
cd ml
pip install -r ../backend/requirements.txt
python train_models.py

# 2. Set up the backend
cd ../backend
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# 3. Create your .env file (copy .env.example, fill in real values)
#    For zero-setup mode, just this one line is enough:
#    DB_MODE=memory

# 4. Start the backend
uvicorn app.main:app --reload --port 8001
```

In a second terminal:
```bash
cd frontend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:API_URL="http://127.0.0.1:8001"
streamlit run dashboard.py
```

Open the Streamlit URL it prints, score a transaction from the sidebar form, and watch it appear in the live feed and charts. Visit `http://127.0.0.1:8001/docs` for interactive API documentation (Swagger UI, auto-generated by FastAPI).

**Note on ports:** the examples above use port `8001` rather than FastAPI's default `8000` — pick any free port on your machine with `--port <number>`, and make sure `API_URL` in the frontend matches whatever you chose.

## Configuration via `.env`

Rather than re-typing `$env:` commands every session, this project loads settings from a `.env` file automatically (via `python-dotenv`). Create `backend/../.env` (i.e. `.env` in the **project root**, one level above `backend/`) with:

```
DB_MODE=memory
# or DB_MODE=astra, plus:
# ASTRA_DB_API_ENDPOINT=https://your-db-id-region.apps.astra.datastax.com
# ASTRA_DB_APPLICATION_TOKEN=AstraCS:your-real-token-here
# ASTRA_KEYSPACE=default_keyspace
```

`config.py` loads this with an explicit path (relative to `config.py`'s own location, not your current directory) and `override=True`, so `.env` always wins over any stale `$env:` variable left over from a previous terminal session — a real problem encountered during development, where an old placeholder token kept silently overriding a correctly-updated `.env` file.

**`.env` is git-ignored** — never commit it. `.env.example` is the safe, shareable template with placeholder values only.

## Running with Real Cassandra (Docker)

```bash
docker compose up --build
```

This starts a real Cassandra 4.1 container, the backend (`DB_MODE=local`), and the frontend, all networked together. First startup takes a minute or two while Cassandra initializes.

## Running with DataStax Astra DB (managed Cassandra) — recommended

1. Create a free database at [astra.datastax.com](https://astra.datastax.com)
2. From the database's **Connect** tab, copy the **API Endpoint** URL (looks like `https://<db-id>-<region>.apps.astra.datastax.com`)
3. Generate an **Application Token** (**Database Administrator** role) — copy the *entire* token immediately; it's only shown once. A real token is long (90-110+ characters) and typically has two colon-separated segments after `AstraCS:` — if what you copied looks short, it was truncated and won't authenticate (this happened during development: a partially-copied token caused persistent `401 Unauthorized` errors that looked like a code bug but were actually just an incomplete copy-paste)
4. Set these in your `.env` (see above)
5. Astra creates a default keyspace automatically (commonly `default_keyspace`) — verify the exact name from your database's page if unsure, since Astra doesn't allow creating keyspaces via CQL the way self-hosted Cassandra does

**Never commit your token to git** — it's covered by `.gitignore`.

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/transactions/score` | POST | Score a transaction in real time, get risk level + action, log it |
| `/transactions/recent?limit=50` | GET | Recent scored transactions, for the live feed |
| `/stats` | GET | Aggregate counts by risk tier, amounts screened/blocked |
| `/health` | GET | Service + model + DB status |
| `/docs` | GET | Interactive Swagger UI |

Example request:
```bash
curl -X POST http://localhost:8001/transactions/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1042, "amount": 25000, "transaction_hour": 2,
    "merchant_category": "ecommerce", "device_type": "web", "location": "Kathmandu",
    "is_new_device": 1, "failed_attempts": 5, "transactions_last_hour": 9,
    "account_age_days": 5
  }'
```

## Risk Tiers & Business Rules

| Risk Level | Trigger | Action |
|---|---|---|
| 🟢 LOW | Blended score below the 95th percentile of normal transactions | **APPROVE** automatically |
| 🟡 MEDIUM | Between the 95th and 99th percentile | **VERIFY** (step-up auth / manual review) |
| 🔴 HIGH | Above the 99th percentile | **BLOCK** |

The blended score is `0.4 × Isolation Forest score + 0.6 × Autoencoder reconstruction error`, both normalized to 0–100.

## Related Work

`notebooks/ieee_fraud_detection.ipynb` is a separate exploration, included here for reference: supervised fraud classification (XGBoost + LightGBM) trained on the real, labeled [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) Kaggle dataset.

It's a different approach from the rest of this repo on purpose — this system uses unsupervised anomaly detection (Isolation Forest + Autoencoder) because real fraud labels don't exist yet for a new wallet provider (a "cold start" problem). The IEEE-CIS notebook shows the alternative: when real labels *are* available, a supervised model like XGBoost typically performs better, since it can learn directly from confirmed fraud cases rather than just flagging statistical outliers.

See also: [Digital-Wallet-Fraud-Detection-eSewa-Khalti-Context-](https://github.com/Rajal22/Digital-Wallet-Fraud-Detection-eSewa-Khalti-Context-), the original exploratory notebook this real-time system is built on.

## Deploying to Google Cloud Platform

**Recommended path:** Astra DB (managed Cassandra, no ops) + Cloud Run (serverless, scales to zero) for both backend and frontend.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

gcloud artifacts repositories create fraud-detection \
  --repository-format=docker --location=asia-south1

gcloud builds submit --tag asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-detection/backend \
  --config=<(echo 'steps: [{name: "gcr.io/cloud-builders/docker", args: ["build","-f","backend/Dockerfile","-t","asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-detection/backend","."]}]
images: ["asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-detection/backend"]')

gcloud secrets create astra-token --data-file=- <<< "AstraCS:your-real-token-here"

gcloud run deploy fraud-backend \
  --image=asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-detection/backend \
  --region=asia-south1 --allow-unauthenticated \
  --set-env-vars=DB_MODE=astra,ASTRA_DB_API_ENDPOINT=https://your-db.apps.astra.datastax.com \
  --set-secrets=ASTRA_DB_APPLICATION_TOKEN=astra-token:latest

gcloud run deploy fraud-frontend \
  --image=asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-detection/frontend \
  --region=asia-south1 --allow-unauthenticated \
  --set-env-vars=API_URL=https://fraud-backend-xxxxx.a.run.app
```

## Suggested 10-Week Build Roadmap

| Week | Focus |
|---|---|
| 1–2 | Data generation, EDA, feature engineering (done) |
| 3 | Isolation Forest + Autoencoder training, threshold tuning (done) |
| 4 | FastAPI backend: scoring endpoint, Pydantic schemas (done) |
| 5 | Astra DB (Cassandra) integration via Data API (done) |
| 6 | Streamlit dashboard: live feed, charts, manual scoring form (done) |
| 7 | Load testing, latency optimization, batch scoring endpoint |
| 8 | GCP deployment: Cloud Run, Artifact Registry, Secret Manager |
| 9 | Monitoring/alerting (Cloud Monitoring, error tracking), CI/CD pipeline |
| 10 | Documentation, demo video, resume write-up, stretch: biometric verification research spike |

## Future Scope

- **Biometric transaction verification** — step-up auth using fingerprint/face match for MEDIUM/HIGH risk transactions, rather than SMS OTP alone
- **Batch scoring endpoint** — score a CSV of transactions at once
- **Model retraining pipeline** — periodically retrain as transaction patterns drift
- **Real fraud labels** — if/when confirmed fraud cases start coming from manual review, use them to build a supervised model alongside the unsupervised one (see `notebooks/ieee_fraud_detection.ipynb`)

## Limitations

- Trained on **synthetic** data with rule-based fraud labels — validate against real transaction patterns before trusting it in production
- Thresholds are static; a real deployment should tune them against actual fraud-loss vs. false-positive costs
- Autoencoder inference happens per-request in this version; at high throughput, batch scoring or a model-serving layer would reduce latency