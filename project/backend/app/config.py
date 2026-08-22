import os
from pathlib import Path

# --- ML artifact location ---
ML_DIR = os.environ.get("ML_DIR", str(Path(__file__).resolve().parent.parent.parent / "ml"))

# --- Database mode ---
# "memory" - in-process list, zero setup, resets on restart (default, good for local dev/demo)
# "local"  - connects to a local/self-hosted Cassandra cluster (e.g. via Docker)
# "astra"  - connects to DataStax Astra DB (managed Cassandra) using a secure connect bundle
DB_MODE = os.environ.get("DB_MODE", "memory").lower()

# --- Local Cassandra settings (DB_MODE=local) ---
CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "127.0.0.1").split(",")
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE", "fraud_detection")

# --- Astra DB settings (DB_MODE=astra) ---
# --- Astra DB settings (DB_MODE=astra) — modern Data API, no CQL driver needed ---
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_KEYSPACE = os.environ.get("ASTRA_KEYSPACE", "default_keyspace")

# --- Risk tiering (business rule thresholds, loaded from trained model config at runtime) ---
LOW_ACTION = "APPROVE"
MEDIUM_ACTION = "VERIFY"
HIGH_ACTION = "BLOCK"
