cd ~/finance-dashboard

cat > pages/20_Autos__Search.py <<'EOF'
import os
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Autos — Search", layout="wide")
st.title("🚗 Autos & Trucks — Undervalued Finder")
st.caption("Sources: OAIKC + GSA. Phase 1: manual watchlist + all-in math + deal scoring.")

# ---- Paths / DB ----
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = os.environ.get("DEALDASH_DB_PATH", str(ROOT / "deals.duckdb"))

# ---- Your defaults / assumptions ----
BUYERS_PREMIUM = 0.10
ADMIN_FEE = 25.0
DEFAULT_TOW = 150.0
DEFAULT_RECON = 500.0

TARGET_MODELS = ["F-150", "Silverado", "Tahoe"]

# ---- DB helpers ----
@st.cache_resource
def db():
    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS autos_watchlist (
          id VARCHAR,
          added_at TIMESTAMP,
          source VARCHAR,
          auction_url VARCHAR,
          lot_url VARCHAR,
          year INTEGER,
          make VARCHAR,
          model VARCHAR,
          trim VARCHAR,
          miles INTEGER,
          title_status VARCHAR,
          current_bid DOUBLE,
          target_resale DOUBLE,
          desired_profit DOUBLE,
          tow_est DOUBLE,
          recon_est DOUBLE,
          notes VARCHAR
        );
        """
    )
    return con

def all_in_from_hammer(hammer: float, tow: float, recon: float) -> float:
    return float(hammer) * (1.0 + BUYERS_PREMIUM) + ADMIN_FEE + float(tow) + float(recon)

def max_hammer(target_resale: float, desired_profit: float, tow: float, recon: float) -> float:
    # target_resale - desired_profit - (admin + tow + recon), then remove buyers premium
    return (float(target_resale) - float(desired_profit) - (ADMIN_FEE + float(tow) + float(recon))) / (1.0 + BUYERS_PREMIUM)

def risk_penalty(notes: str) -> float:
    n = (notes or "").lower()
    penalty = 0.0
    if "no keys" in n or "unknown keys" in n:
        penalty += 400.0
    if "unknown run" in n or "unknown drive" in n or "won't start" in n or "wont start" in n:
        penalty += 400.0
    if any(k in n for k in ["airbag", "frame", "flood", "hail", "totaled", "salvage", "rebuilt", "damage"]):
        penalty += 800.0
    if "unknown miles" in n or "miles unknown" in n:
        penalty += 300.0
    return penalty

# ---- Sidebar filters ----
st.sidebar.header("Filters")
model_filter = st.sidebar.multiselect("Target models", TARGET_MODELS, default=TARGET_MODELS)
max_bid_filter = st.sidebar.number_input("Max current bid (optional)", value=15000.0, step=500.0)

# ---- Add lot form ----
st.subheader("Add a lot to watchlist")

with st.form("add_lot"):
    c1, c2, c3 = st.columns(3)

    with c1:
        source = st.selectbox("Source", ["OAIKC", "GSA"])
        auction_url = st.text_input("Auction URL", value="https://bids.oaikc.com/auctions/45028-2025-12-16-kc-tow-lot")
        lot_url = st.text_input("Lot URL", value="")
        title_status = st.selectbox("Title status", ["clean", "unknown", "salvage/rebuilt"], index=0)

    with c2:
        year = st.number_input("Year", min_value=1980, max_value=2035, value=2016, step=1)
        make = st.text_input("Make", value="Ford")
        model = st.text_input("Model", value="F-150")
        trim = st.text_input("Trim (optional)", value="")
        miles = st.number_input("Miles (0 if unknown)", min_value=0, value=0, step=1000)

    with c3:
        current_bid = st.number_input("Current bid / expected hammer", min_value=0.0, value=12000.0, step=250.0)
        target_resale = st.number_input("Target resale (retail)", min_value=0.0, value=18500.0, step=500.0)
        desired_profit = st.number_input("Desired profit", min_value=0.0, value=2500.0, step=250.0)
        tow_est = st.number_input("Tow estimate", min_value=0.0, value=DEFAULT_TOW, step=25.0)
        recon_est = st.number_input("Recon estimate", min_value=0.0, value=DEFAULT_RECON, step=100.0)

    notes = st.text_input("Notes / risk flags", value="", placeholder="no keys, unknown run/drive, damage keywords, etc.")

    submitted = st.form_submit_button("Add to watchlist")
    if submitted:
        if title_status != "clean":
            st.error("Clean title only: this lot is blocked by your rule.")
        else:
            uid = f"{int(datetime.utcnow().timestamp())}-{abs(hash(lot_url or (str(year)+make+model+trim))) }"
            con = db()
            con.execute(
                """
                INSERT INTO autos_watchlist VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    uid,
                    datetime.utcnow(),
                    source,
                    auction_url.strip(),
                    lot_url.strip(),
                    int(year),
                    make.strip(),
                    model.strip(),
                    trim.strip(),
                    int(miles) if miles and int(miles) > 0 else None,
                    title_status,
                    float(current_bid),
                    float(target_resale),
                    float(desired_profit),
                    float(tow_est),
                    float(recon_est),
                    notes.strip(),
                ],
            )
            st.success("Added to watchlist.")

# ---- Load / compute ----
st.subheader("Watchlist")

con = db()
df = con.execute("SELECT * FROM autos_watchlist ORDER BY added_at DESC").df()

if df.empty:
    st.info("No watched lots yet.")
else:
    # Filters
    if model_filter:
        df = df[df["model"].fillna("").isin(model_filter)]
    if max_bid_filter:
        df = df[df["current_bid"].fillna(0.0) <= float(max_bid_filter)]

    rows = []
    for _, r in df.iterrows():
        hammer = float(r["current_bid"] or 0.0)
        target = float(r["target_resale"] or 0.0)
        desired = float(r["desired_profit"] or 0.0)
        tow = float(r["tow_est"] or DEFAULT_TOW)
        recon = float(r["recon_est"] or DEFAULT_RECON)
        notes_val = r["notes"] or ""

        all_in = all_in_from_hammer(hammer, tow, recon)
        profit = target - all_in
        roi = (profit / all_in) if all_in > 0 else 0.0
        penalty = risk_penalty(notes_val)
        score = profit - penalty
        mh = max_hammer(target, desired, tow, recon)

        vehicle_label = "{} {} {} {}".format(
            int(r["year"]) if pd.notna(r["year"]) else "",
            (r["make"] or "").strip(),
            (r["model"] or "").strip(),
            (r["trim"] or "").strip(),
        ).strip()

        rows.append(
            {
                "added_at": r["added_at"],
                "vehicle": vehicle_label,
                "source": r["source"],
                "current_bid": hammer,
                "max_hammer_bid": mh,
                "all_in_est": all_in,
                "target_resale": target,
                "profit_est": profit,
                "roi_est": roi,
                "risk_penalty": penalty,
                "deal_score": score,
                "lot_url": r["lot_url"],
                "notes": notes_val,
            }
        )

    out = pd.DataFrame(rows).sort_values("deal_score", ascending=False)
    st.dataframe(out, use_container_width=True)

    st.caption("All-in (OAIKC default) = hammer*1.10 + $25 + tow + recon. Score = profit minus risk penalties.")
EOF
