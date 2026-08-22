import os
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Fraud Detection — Live Monitor",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Digital Wallet Fraud Detection — Live Monitor")
st.caption("Real-time transaction risk monitoring for Nepal digital wallets (eSewa/Khalti-style)")

# ---------------------------------------------------------------------
# Sidebar: manual transaction scoring form
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Score a Transaction")
    st.caption("Simulate a live transaction and see it scored in real time.")

    with st.form("score_form"):
        user_id = st.number_input("User ID", min_value=1, value=1042)
        amount = st.number_input("Amount (NPR)", min_value=1.0, value=1500.0, step=100.0)
        transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
        merchant_category = st.selectbox(
            "Merchant Category",
            ["grocery", "restaurant", "utility_bill", "mobile_topup", "ecommerce",
             "travel", "electronics", "fuel", "education", "healthcare"],
        )
        device_type = st.selectbox("Device Type", ["android", "ios", "web"])
        location = st.selectbox(
            "Location",
            ["Kathmandu", "Pokhara", "Lalitpur", "Bhaktapur", "Biratnagar",
             "Birgunj", "Dharan", "Butwal", "Nepalgunj", "Hetauda"],
        )
        is_new_device = st.checkbox("New / unrecognized device")
        failed_attempts = st.slider("Failed login attempts (last session)", 0, 10, 0)
        transactions_last_hour = st.slider("Transactions in the last hour", 0, 20, 1)
        account_age_days = st.number_input("Account age (days)", min_value=0, value=400)

        submitted = st.form_submit_button("Score Transaction", use_container_width=True)

    if submitted:
        payload = {
            "user_id": int(user_id),
            "amount": float(amount),
            "transaction_hour": int(transaction_hour),
            "merchant_category": merchant_category,
            "device_type": device_type,
            "location": location,
            "is_new_device": int(is_new_device),
            "failed_attempts": int(failed_attempts),
            "transactions_last_hour": int(transactions_last_hour),
            "account_age_days": int(account_age_days),
        }
        try:
            resp = requests.post(f"{API_URL}/transactions/score", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Scoring failed: {e}")

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(r["risk_level"], "⚪")
        st.markdown(f"### {color} {r['risk_level']} — {r['action']}")
        st.metric("Risk Score", f"{r['risk_score']:.1f} / 100")
        c1, c2 = st.columns(2)
        c1.metric("Isolation Forest", f"{r['isolation_score']:.1f}")
        c2.metric("Autoencoder", f"{r['autoencoder_score']:.1f}")

# ---------------------------------------------------------------------
# Main area: live stats + feed
# ---------------------------------------------------------------------
auto_refresh = st.toggle("Auto-refresh every 5s", value=False)

try:
    stats = requests.get(f"{API_URL}/stats", timeout=10).json()
    recent = requests.get(f"{API_URL}/transactions/recent?limit=100", timeout=10).json()
except Exception as e:
    st.error(f"Could not reach the fraud detection API at {API_URL}. Is the backend running? ({e})")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions Screened", stats["total_transactions"])
col2.metric("🔴 High Risk (Blocked)", stats["high_count"])
col3.metric("💰 Total Amount Screened (NPR)", f"{stats['total_amount_screened']:,.0f}")
col4.metric("🚫 Amount Blocked (NPR)", f"{stats['blocked_amount']:,.0f}")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Risk Distribution")
    if stats["total_transactions"] > 0:
        dist_df = pd.DataFrame({
            "Risk Level": ["LOW", "MEDIUM", "HIGH"],
            "Count": [stats["low_count"], stats["medium_count"], stats["high_count"]],
        })
        fig = px.pie(
            dist_df, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"},
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions scored yet — use the form on the left to score one.")

with right:
    st.subheader("Risk Score Over Time")
    if recent:
        df = pd.DataFrame(recent)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["created_at"], y=df["risk_score"], mode="markers+lines",
            marker=dict(
                color=df["risk_level"].map({"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}),
                size=8,
            ),
            line=dict(color="rgba(150,150,150,0.3)"),
            name="Risk Score",
        ))
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4)
        fig.update_layout(yaxis_title="Risk Score (0-100)", xaxis_title="Time", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions yet.")

st.divider()
st.subheader("Live Transaction Feed")

if recent:
    feed_df = pd.DataFrame(recent)
    feed_df["created_at"] = pd.to_datetime(feed_df["created_at"]).dt.strftime("%H:%M:%S")

    def highlight_risk(row):
        color = {"LOW": "#eafaf1", "MEDIUM": "#fef5e7", "HIGH": "#fdedec"}.get(row["risk_level"], "")
        return [f"background-color: {color}"] * len(row)

    display_cols = ["created_at", "user_id", "amount", "risk_score", "risk_level", "action"]
    display_cols = [c for c in display_cols if c in feed_df.columns]
    st.dataframe(
        feed_df[display_cols].style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=400,
    )
else:
    st.info("No transactions yet — score one from the sidebar to see it appear here.")

if auto_refresh:
    time.sleep(5)
    st.rerun()
