from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    user_id: int = Field(..., examples=[1042])
    amount: float = Field(..., gt=0, examples=[2500.0])
    transaction_hour: int = Field(..., ge=0, le=23, examples=[14])
    merchant_category: str = Field(..., examples=["ecommerce"])
    device_type: str = Field(..., examples=["android"])
    location: str = Field(..., examples=["Kathmandu"])
    is_new_device: int = Field(0, ge=0, le=1)
    failed_attempts: int = Field(0, ge=0, le=20)
    transactions_last_hour: int = Field(1, ge=0, le=100)
    account_age_days: int = Field(..., ge=0, examples=[365])


class TransactionResponse(BaseModel):
    transaction_id: str
    user_id: int
    amount: float
    risk_score: float
    isolation_score: float
    autoencoder_score: float
    risk_level: str
    action: str
    created_at: datetime


class StatsResponse(BaseModel):
    total_transactions: int
    low_count: int
    medium_count: int
    high_count: int
    total_amount_screened: float
    blocked_amount: float


class HealthResponse(BaseModel):
    status: str
    db_mode: str
    models_loaded: bool
