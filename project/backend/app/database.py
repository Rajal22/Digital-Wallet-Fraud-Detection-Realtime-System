import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from . import config


class InMemoryStore:
    """Zero-setup fallback store. Good for local dev/demo without any
    Cassandra cluster running. Data resets whenever the process restarts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: List[dict] = []

    def insert_transaction(self, row: dict):
        with self._lock:
            self._rows.append(row)

    def get_recent(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(reversed(self._rows[-limit:]))

    def get_stats(self) -> dict:
        with self._lock:
            rows = self._rows
        total = len(rows)
        low = sum(1 for r in rows if r["risk_level"] == "LOW")
        medium = sum(1 for r in rows if r["risk_level"] == "MEDIUM")
        high = sum(1 for r in rows if r["risk_level"] == "HIGH")
        total_amount = sum(r["amount"] for r in rows)
        blocked_amount = sum(r["amount"] for r in rows if r["action"] == "BLOCK")
        return {
            "total_transactions": total,
            "low_count": low,
            "medium_count": medium,
            "high_count": high,
            "total_amount_screened": round(total_amount, 2),
            "blocked_amount": round(blocked_amount, 2),
        }


class AstraDataAPIStore:
    """Storage using DataStax Astra DB's modern Data API (astrapy) — a
    document/collection interface over Cassandra, no CQL driver or secure
    connect bundle needed."""

    def __init__(self):
        from astrapy import DataAPIClient

        client = DataAPIClient(config.ASTRA_DB_APPLICATION_TOKEN)
        self.db = client.get_database(
            config.ASTRA_DB_API_ENDPOINT,
            keyspace=config.ASTRA_KEYSPACE,
        )
        try:
            self.collection = self.db.create_collection("transactions")
        except Exception:
            self.collection = self.db.get_collection("transactions")

    def insert_transaction(self, row: dict):
        doc = dict(row)
        doc["_id"] = doc.pop("transaction_id")
        doc["created_at"] = doc["created_at"].isoformat()
        self.collection.insert_one(doc)

    def get_recent(self, limit: int = 50) -> List[dict]:
        cursor = self.collection.find({}, sort={"created_at": -1}, limit=limit)
        results = []
        for doc in cursor:
            doc["transaction_id"] = doc.pop("_id")
            results.append(doc)
        return results

    def get_stats(self) -> dict:
        docs = list(self.collection.find({}))
        total = len(docs)
        low = sum(1 for d in docs if d["risk_level"] == "LOW")
        medium = sum(1 for d in docs if d["risk_level"] == "MEDIUM")
        high = sum(1 for d in docs if d["risk_level"] == "HIGH")
        total_amount = sum(d["amount"] for d in docs)
        blocked_amount = sum(d["amount"] for d in docs if d["action"] == "BLOCK")
        return {
            "total_transactions": total,
            "low_count": low,
            "medium_count": medium,
            "high_count": high,
            "total_amount_screened": round(total_amount, 2),
            "blocked_amount": round(blocked_amount, 2),
        }



class CassandraStore:
    """Real Cassandra storage — works against either a local self-hosted
    cluster (DB_MODE=local) or DataStax Astra DB (DB_MODE=astra), which is
    just managed Cassandra with a secure connect bundle for auth."""



    def __init__(self):
        from cassandra.cluster import Cluster

        self.cluster = Cluster(config.CASSANDRA_HOSTS, port=config.CASSANDRA_PORT)
        self.keyspace = config.CASSANDRA_KEYSPACE
        self.session = self.cluster.connect()
        self._setup_schema()

    def _setup_schema(self):
        self.session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
        """)

        self.session.set_keyspace(self.keyspace)

   

        self.session.set_keyspace(self.keyspace)

        self.session.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id text PRIMARY KEY,
                user_id int,
                amount double,
                merchant_category text,
                device_type text,
                location text,
                risk_score double,
                isolation_score double,
                autoencoder_score double,
                risk_level text,
                action text,
                created_at timestamp
            )
        """)

        # Secondary table ordered for "recent transactions" queries — Cassandra
        # doesn't support ORDER BY on arbitrary columns, so we keep a
        # time-bucketed table for feed queries. Single bucket is fine for a demo;
        # a production system would bucket by hour/day to keep partitions small.
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS transactions_by_time (
                bucket text,
                created_at timestamp,
                transaction_id text,
                user_id int,
                amount double,
                risk_score double,
                isolation_score double,
                autoencoder_score double,
                risk_level text,
                action text,
                PRIMARY KEY (bucket, created_at, transaction_id)
            ) WITH CLUSTERING ORDER BY (created_at DESC)
        """)

    def insert_transaction(self, row: dict):
        self.session.execute("""
            INSERT INTO transactions (
                transaction_id, user_id, amount, merchant_category, device_type,
                location, risk_score, isolation_score, autoencoder_score,
                risk_level, action, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row["transaction_id"], row["user_id"], row["amount"],
            row["merchant_category"], row["device_type"], row["location"],
            row["risk_score"], row["isolation_score"], row["autoencoder_score"],
            row["risk_level"], row["action"], row["created_at"],
        ))

        self.session.execute("""
            INSERT INTO transactions_by_time (
                bucket, created_at, transaction_id, user_id, amount,
                risk_score, isolation_score, autoencoder_score, risk_level, action
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            "all", row["created_at"], row["transaction_id"], row["user_id"],
            row["amount"], row["risk_score"], row["isolation_score"],
            row["autoencoder_score"], row["risk_level"], row["action"],
        ))

    def get_recent(self, limit: int = 50) -> List[dict]:
        rows = self.session.execute(
            "SELECT * FROM transactions_by_time WHERE bucket = %s LIMIT %s",
            ("all", limit),
        )
        return [dict(r._asdict()) for r in rows]

    def get_stats(self) -> dict:
        # For a demo-scale table, a full scan is fine. At real production
        # volume this would instead maintain running counters (e.g. a
        # separate counter table updated on each insert).
        rows = list(self.session.execute("SELECT amount, risk_level, action FROM transactions"))
        total = len(rows)
        low = sum(1 for r in rows if r.risk_level == "LOW")
        medium = sum(1 for r in rows if r.risk_level == "MEDIUM")
        high = sum(1 for r in rows if r.risk_level == "HIGH")
        total_amount = sum(r.amount for r in rows)
        blocked_amount = sum(r.amount for r in rows if r.action == "BLOCK")
        return {
            "total_transactions": total,
            "low_count": low,
            "medium_count": medium,
            "high_count": high,
            "total_amount_screened": round(total_amount, 2),
            "blocked_amount": round(blocked_amount, 2),
        }


_store = None


def get_store():
    global _store
    if _store is None:
        if config.DB_MODE == "astra":
            _store = AstraDataAPIStore()
        elif config.DB_MODE == "local":
            _store = CassandraStore()
        else:
            _store = InMemoryStore()
    return _store


def make_transaction_row(txn: dict, score_result: dict) -> dict:
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": txn["user_id"],
        "amount": txn["amount"],
        "merchant_category": txn["merchant_category"],
        "device_type": txn["device_type"],
        "location": txn["location"],
        "created_at": datetime.now(timezone.utc),
        **score_result,
    }
