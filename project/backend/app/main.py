from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .models import TransactionRequest, TransactionResponse, StatsResponse, HealthResponse
from .ml_models import get_scorer
from .database import get_store, make_transaction_row

app = FastAPI(
    title="Digital Wallet Fraud Detection API",
    description="Real-time fraud scoring for Nepal digital wallet transactions "
                 "(eSewa/Khalti-style) using Isolation Forest + Autoencoder.",
    version="1.0.0",
)

# Allow the Streamlit dashboard (and any local dev frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # Load models and connect to the store once, at startup, not per-request
    get_scorer()
    get_store()


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_scorer()
        models_loaded = True
    except Exception:
        models_loaded = False
    return HealthResponse(status="ok", db_mode=config.DB_MODE, models_loaded=models_loaded)


@app.post("/transactions/score", response_model=TransactionResponse)
def score_transaction(txn: TransactionRequest):
    """Score a transaction in real time and log the result."""
    try:
        scorer = get_scorer()
        result = scorer.score(txn.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    row = make_transaction_row(txn.model_dump(), result)

    try:
        get_store().insert_transaction(row)
    except Exception as e:
        # Don't fail the request just because logging failed — the caller
        # still needs their fraud decision. Log and move on.
        print(f"Warning: failed to persist transaction: {e}")

    return TransactionResponse(**row)


@app.get("/transactions/recent", response_model=list[TransactionResponse])
def recent_transactions(limit: int = 50):
    rows = get_store().get_recent(limit=limit)
    # Cassandra rows won't have merchant_category/device_type/location in the
    # transactions_by_time table (kept lean for feed queries) — fill blanks
    # so the response model doesn't choke. The dashboard mainly cares about
    # score/level/action/timestamp for the live feed.
    for r in rows:
        r.setdefault("merchant_category", "")
        r.setdefault("device_type", "")
        r.setdefault("location", "")
    return [TransactionResponse(**r) for r in rows]


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(**get_store().get_stats())


@app.get("/")
def root():
    return {
        "service": "Digital Wallet Fraud Detection API",
        "docs": "/docs",
        "db_mode": config.DB_MODE,
    }
