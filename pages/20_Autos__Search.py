

import streamlit as st
import duckdb
from datetime import datetime
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Autos — Search", layout="wide")
st.title("🚗 Autos & Trucks — Undervalued Finder")

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(ROOT / "autos.duckdb")  # or /var/lib/dealdash/deals.duckdb on EC2

# --- Defaults (your assumptions) ---
BUYERS_PREMIUM = 0.10
ADMIN_FEE = 25
TOW_EST = 150
RECON_EST = 500

TARGET_MODELS = ["F-150", "Silverado", "Tahoe"]

@st.cache_resource
def db():
    con = duckdb.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS autos_watchlist (
      id VARCHAR,
      added_at TIMESTAMP,
      source VARCHAR,
      auction_url VARCHAR,
      lot_url VARCHAR,
      vehicle VARCHAR,
      year INTEGER,
      make VARCHAR,
      model VARCHAR,
      miles INTEGER,
      title_status VARCHAR,
      current_bid DOUBLE,
      target_resale DOUBLE,
      desired_profit DOUBLE,
      tow_est DOUBLE,
      recon_est DOUBLE,
      risk_flags VARCHAR
    );
    """)
    return con

def all_in_from_hammer(hammer: float, tow: float, recon: float) -> float:
    return hammer * (1.0 + BUYERS_PREMIUM) + ADMIN_FEE + tow + recon

def max_hammer(target_resale: float, desired_profit: float, tow: float, recon: float) -> float:
    # target_resale - desired_profit - (admin+tow+recon) then remove buyers premium
    return (target_resale - desired_profit - (ADMIN_FEE + tow + recon)) / (1.0 + BUYERS_PREMIUM)

def compute_score(target_resale: float, hammer: float, tow: float, recon: float, risk_flags: str) -> dict:
    all_in = all_in_from_hammer(hammer, tow, recon)
    profit = target_resale - all_in
    roi = profit / all_in if all_in > 0 else 0.0

    penalty = 0
    flags = (risk_flags or "").lower()
    if "no keys" in flags or "unknown keys" in flags:
        penalty += 400
    if "unknown run" in flags or "unknown drive" in flags:
        penalty += 400
    if any(k in flags for k in ["airbag", "frame", "flood", "hail", "totaled", "salvage", "rebuilt", "damage"]):
        penalty += 800
    if "miles unknown" in flags or "unknown miles" in flags:
        penalty += 300

    score = profit - penalty
    return {"all_in": all_in, "profit": profit, "roi": roi, "penalty": penalty, "score": score}

# --- Sidebar filters ---
st.sidebar.header("Filters")
model_filter = st.sidebar.multiselect("Target models", TARGET_MODELS, default=TARGET_MODELS)
max_bid_filter = st.sidebar.number_input("Max current bid (optional)", value=15000.0, step=500.0)

# --- Add lot form ---
st.subheader("Add a lot to watchlist")
with st.form("add_lot"):
    c1, c2, c3 = st.columns(3)
    with c1:
        source = st.selectbox("Source", ["OAIKC", "GSA"])
        auction_url = st.text_input("Auction URL", value="https://bids.oaikc.com/auctions/45028-2025-12-16-kc-tow-lot")
        lot_url = st.text_input("Lot URL")
        title_status = st.selectbox("Title status", ["clean", "unknown", "salvage/rebuilt"], index=0)
    with c2:
        vehicle = st.text_input("Vehicle (free text)", placeholder="2016 Ford F-150 XLT ...")
        year = st.number_input("Year", min_value=1980, max_value=2030, value=2016, step=1)
        make = st.text_input("Make", value="Ford")
        model = st.text_input("Model", value="F-150")
        miles = st.number_input("Miles (0 if unknown)", min_value=0, value=0, step=1000)
    with c3:
        current_bid = st.number_input("Current bid / expected hammer", min_value=0.0, value=12000.0, step=250.0)
        target_resale = st.number_input("Target resale (retail)", min_value=0.0, value=18500.0, step=500.0)
        desired_profit = st.number_input("Desired profit", min_value=0.0, value=2500.0, step=250.0)
        tow_est = st.number_input("Tow estimate", min_value=0.0, value=float(TOW_EST), step=25.0)
        recon_est = st.number_input("Recon estimate", min_value=0.0, value=float(RECON_EST), step=100.0)

    risk_flags = st.text_input("Risk flags (notes)", placeholder="no keys, unknown run/drive, damage...")

    submitted = st.form_submit_button("Add to watchlist")
    if submitted:
        if title_status != "clean":
            st.error("Clean title only: this lot is blocked by your rule.")
        else:
            # basic target model gating
            if model_filter and (model.strip() not in model_filter):
                st.warning("Model isn’t in your target list, but adding anyway.")

            uid = f"{int(datetime.utcnow().timestamp())}-{abs(hash(lot_url or vehicle))}"
            con = db()
            con.execute(
                "INSERT INTO autos_watchlist VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    uid,
                    datetime.utcnow(),
                    source,
                    auction_url,
                    lot_url,
                    vehicle,
                    int(year),
                    make,
                    model,
                    int(miles) if miles else None,
                    title_status,
                    float(current_bid),
                    float(target_resale),
                    float(desired_profit),
                    float(tow_est),
                    float(recon_est),
                    risk_flags,
                ],
            )
            st.success("Added.")

# --- Display watchlist ---
st.subheader("Watchlist")

con = db()
df = con.execute("SELECT * FROM autos_watchlist ORDER BY added_at DESC").df()

if df.empty:
    st.info("No watched lots yet.")
else:
    # Apply filter logic
    if model_filter:
        df = df[df["model"].fillna("").isin(model_filter)]
    if max_bid_filter:
        df = df[df["current_bid"].fillna(0) <= max_bid_filter]

    # Compute deal math
    rows = []
    for _, r in df.iterrows():
        hammer = float(r["current_bid"] or 0)
        target = float(r["target_resale"] or 0)
        tow = float(r["tow_est"] or TOW_EST)
        recon = float(r["recon_est"] or RECON_EST)

        metrics = compute_score(target, hammer, tow, recon, r.get("risk_flags", ""))
        mh = max_hammer(target, float(r["desired_profit"] or 0), tow, recon)

        rows.append({
            "added_at": r["added_at"],
            "vehicle": r["vehicle"] or f'{r["year"]} {
