import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime

import yfinance as yf

DATA_DIR = pathlib.Path("data")
INDEX_FILES = {
    "S&P 500": DATA_DIR / "sp500.txt",
    "Dow 30": DATA_DIR / "dow30.txt",
    "Nasdaq 100": DATA_DIR / "nasdaq100.txt",
}


from core.data import load_price_data
from core.signals import (
    identify_breakouts,
    compute_strong_trend_mask,
    compute_consolidation_mask,
    identify_bull_flags,
)
from core.backtest import backtest_trades
from core.stats import compute_stats


# -----------------------------
# Data loading wrapper
# -----------------------------
@st.cache_data(show_spinner=False)
def load_price_data_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_price_data(ticker, start, end)


# -----------------------------
# Index → universe helper
# -----------------------------
def _load_index_file(path: pathlib.Path):
    """
    Load tickers from a text file, one per line.
    Lines starting with # or blank lines are ignored.
    """
    if not path.exists():
        return []

    tickers = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    return tickers


def get_index_universe(index_name: str):
    """
    Return a list of tickers for a given index name.

    First tries to load from data/sp500.txt or data/dow30.txt.
    If the file does not exist or is empty, falls back to a small
    built-in sample list so the app still works.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if index_name == "S&P 500":
        tickers = _load_index_file(INDEX_FILES["S&P 500"])
        if tickers:
            return tickers
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA"]

    if index_name == "Dow 30":
        tickers = _load_index_file(INDEX_FILES["Dow 30"])
        if tickers:
            return tickers
        return ["AAPL", "MSFT", "JPM", "GS", "KO", "DIS", "BA"]

    if index_name == "Nasdaq 100":
        tickers = _load_index_file(INDEX_FILES["Nasdaq 100"])
        if tickers:
            return tickers
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA", "ADBE", "NFLX"]

    return []



# -----------------------------
# Per-ticker model run
# -----------------------------
def run_model_on_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    breakout_lookback: int,
    breakout_buffer_pct: float,
    max_flag_pullback_pct: float,
    min_flag_len: int,
    max_flag_len: int,
    min_trend_months: int,
    min_trend_return_pct: float,
    min_consol_months: int,
    max_consol_height_pct: float,
    starting_equity: float,
    risk_per_trade_pct: float,
    trailing_stop_pct: float,
    recent_signal_lookback_days: int,
):
    """
    Run the full pipeline on a single ticker and return a stats dict.

    Adds:
      - has_recent_signal: whether a bull-flag entry appeared in the last N days
      - last_signal_date: most recent entry_date (if any)
      - num_trades, CAGR, etc. from backtest for context
    """
    df = load_price_data_cached(ticker, start_date, end_date)
    if df.empty:
        return None

    # --- Signal generation: trend + consolidation + breakout + bull flags ---
    breakout_flags = identify_breakouts(df, breakout_lookback, breakout_buffer_pct)
    strong_trend_mask = compute_strong_trend_mask(
        df, min_trend_months, min_trend_return_pct
    )
    consolidation_mask = compute_consolidation_mask(
        df, min_consol_months, max_consol_height_pct
    )

    filtered_breakouts = breakout_flags & strong_trend_mask & consolidation_mask

    trades = identify_bull_flags(
        df,
        filtered_breakouts,
        max_flag_pullback_pct,
        min_flag_len,
        max_flag_len,
    )

    # --- Recent signal detection ---
    has_recent_signal = False
    last_signal_date = None

    if trades:
        all_dates = [pd.to_datetime(t["entry_date"]) for t in trades]
        last_signal_date = max(all_dates)

        cutoff = df.index[-1] - pd.Timedelta(days=recent_signal_lookback_days)
        has_recent_signal = any(d >= cutoff for d in all_dates)

    # --- Backtest (for context) ---
    if not trades:
        equity_curve = pd.Series(
            data=starting_equity, index=df.index, dtype=float
        )
        trade_results = []
    else:
        equity_curve, trade_results = backtest_trades(
            df,
            trades,
            starting_equity,
            risk_per_trade_pct,
            trailing_stop_pct,
        )

    stats = compute_stats(equity_curve, trade_results, starting_equity)
    stats["num_trades"] = stats.get("num_trades", 0)
    stats["ticker"] = ticker
    stats["has_recent_signal"] = has_recent_signal
    stats["last_signal_date"] = last_signal_date

    return stats


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Bull-Flag Multi-Ticker Scanner", layout="wide")

    # Zaheer-style defaults (same spirit as app.py)
    ZAHEER_DEFAULT = {
        "breakout_lookback": 126,
        "breakout_buffer_pct": 0.5,
        "max_flag_pullback_pct": 15.0,
        "min_flag_len": 4,
        "max_flag_len": 25,
        "min_trend_months": 18,
        "min_trend_return_pct": 120.0,
        "min_consol_months": 3,
        "max_consol_height_pct": 40.0,
        "starting_equity": 1_000.0,
        "risk_per_trade_pct": 1.0,
        "trailing_stop_pct": 15.0,
        "recent_signal_lookback_days": 10,
        "min_confidence_cagr": -20.0,
        "min_trades": 0,
    }
    # More permissive than Zaheer defaults to surface additional tickers
    LOOSER_PRESET = {
        "breakout_lookback": 90,
        "breakout_buffer_pct": 0.1,
        "max_flag_pullback_pct": 20.0,
        "min_flag_len": 3,
        "max_flag_len": 40,
        "min_trend_months": 12,
        "min_trend_return_pct": 60.0,
        "min_consol_months": 2,
        "max_consol_height_pct": 60.0,
        "starting_equity": 1_000.0,
        "risk_per_trade_pct": 1.0,
        "trailing_stop_pct": 20.0,
        "recent_signal_lookback_days": 15,
        "min_confidence_cagr": -20.0,
        "min_trades": 0,
    }

    # Initialize session state defaults
    for k, v in ZAHEER_DEFAULT.items():
        st.session_state.setdefault(k, v)

    st.title("Bull-Flag Multi-Ticker Scanner")
    st.markdown(
        "Scan an index or custom list of tickers with the same Zaheer-style bull-flag model.\n\n"
        "- **Signal mode:** focus on tickers with a **recent bull-flag setup**.\n"
        "- **Backtest stats:** used as *context* (historical behavior), not the primary gate."
    )

    # ----- Sidebar: Universe & Dates -----
    st.sidebar.header("Universe & Dates")

    universe_mode = st.sidebar.radio(
        "Universe source",
        ["Manual list", "Index"],
        index=0,
    )

    index_choice = None
    tickers_text = ""

    if universe_mode == "Manual list":
        tickers_text = st.sidebar.text_area(
            "Tickers (comma / space separated)",
            value="AAPL, MSFT, NVDA, TSLA, META, AMZN, GOOG",
            height=80,
        )
    else:
        index_choice = st.sidebar.selectbox(
            "Index to scan",
            ["S&P 500", "Dow 30", "Nasdaq 100"],
        )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.sidebar.date_input("Start date", value=datetime(2014, 1, 1))
    with col2:
        end_date = st.sidebar.date_input("End date", value=datetime.today())

    # Preset button
    if st.sidebar.button("Apply Zaheer Preset"):
        for k, v in ZAHEER_DEFAULT.items():
            st.session_state[k] = v
    if st.sidebar.button("Apply Looser Preset"):
        for k, v in LOOSER_PRESET.items():
            st.session_state[k] = v
    st.sidebar.caption(
        "Zaheer preset = stricter trend/consolidation; Looser preset = surfaces more tickers."
    )

    # ----- Sidebar: model parameters -----
    st.sidebar.header("Breakout & Bull Flag Settings")
    c_breakout, c_buffer = st.sidebar.columns(2)
    with c_breakout:
        breakout_lookback = st.sidebar.slider(
            "Breakout lookback (days)",
            min_value=20,
            max_value=252,
            value=st.session_state["breakout_lookback"],
            step=5,
            key="breakout_lookback",
            help="How far back to look for the prior high; 126 ~ 6 months of trading days.",
        )
    with c_buffer:
        breakout_buffer_pct = st.sidebar.slider(
            "Breakout buffer (%)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state["breakout_buffer_pct"],
            step=0.1,
            key="breakout_buffer_pct",
            help="Extra % above the prior high required to call it a breakout.",
        )

    c_pullback, c_len = st.sidebar.columns(2)
    with c_pullback:
        max_flag_pullback_pct = st.sidebar.slider(
            "Max flag pullback (%)",
            min_value=2.0,
            max_value=30.0,
            value=st.session_state["max_flag_pullback_pct"],
            step=1.0,
            key="max_flag_pullback_pct",
            help="How deep the flag can retrace from the breakout close.",
        )
    with c_len:
        min_flag_len = st.sidebar.slider(
            "Min flag length (days)",
            min_value=3,
            max_value=30,
            value=st.session_state["min_flag_len"],
            step=1,
            key="min_flag_len",
            help="Shortest acceptable flag/base.",
        )
    max_flag_len = st.sidebar.slider(
        "Max flag length (days)",
        min_value=5,
        max_value=60,
        value=st.session_state["max_flag_len"],
        step=1,
        key="max_flag_len",
        help="Longest acceptable flag/base.",
    )

    st.sidebar.header("Prior Trend Filter")
    c_trend_len, c_trend_ret = st.sidebar.columns(2)
    with c_trend_len:
        min_trend_months = st.sidebar.slider(
            "Min trend length (months)",
            min_value=6,
            max_value=36,
            value=st.session_state["min_trend_months"],
            step=1,
            key="min_trend_months",
            help="How many months of prior uptrend to require.",
        )
    with c_trend_ret:
        min_trend_return_pct = st.sidebar.slider(
            "Min trend return (%)",
            min_value=20.0,
            max_value=300.0,
            value=st.session_state["min_trend_return_pct"],
            step=5.0,
            key="min_trend_return_pct",
            help="Minimum % gain over that trend window.",
        )

    st.sidebar.header("Consolidation Filter")
    c_consol_len, c_consol_ht = st.sidebar.columns(2)
    with c_consol_len:
        min_consol_months = st.sidebar.slider(
            "Min consolidation (months)",
            min_value=2,
            max_value=24,
            value=st.session_state["min_consol_months"],
            step=1,
            key="min_consol_months",
            help="Minimum duration of the base/range.",
        )
    with c_consol_ht:
        max_consol_height_pct = st.sidebar.slider(
            "Max consolidation height (%)",
            min_value=5.0,
            max_value=60.0,
            value=st.session_state["max_consol_height_pct"],
            step=1.0,
            key="max_consol_height_pct",
            help="Max % top-to-bottom height of the range.",
        )

    with st.sidebar.expander("Risk Management", expanded=False):
        starting_equity = st.number_input(
            "Starting equity ($) per ticker",
            min_value=1000.0,
            max_value=1_000_000.0,
            value=st.session_state["starting_equity"],
            step=1000.0,
            key="starting_equity",
            help="Used for position sizing and equity curves.",
        )
        risk_per_trade_pct = st.slider(
            "Risk per trade (% of equity)",
            min_value=0.25,
            max_value=5.0,
            value=st.session_state["risk_per_trade_pct"],
            step=0.25,
            key="risk_per_trade_pct",
            help="Percent of equity risked per trade based on stop distance.",
        )
        trailing_stop_pct = st.slider(
            "Trailing stop distance (%)",
            min_value=2.0,
            max_value=40.0,
            value=st.session_state["trailing_stop_pct"],
            step=1.0,
            key="trailing_stop_pct",
            help="Stop follows highest close by this % to exit winners.",
        )

    st.sidebar.header("Recent Signal Filter")
    recent_signal_lookback_days = st.sidebar.slider(
        "Lookback window for 'recent' signal (days)",
        min_value=1,
        max_value=60,
        value=st.session_state["recent_signal_lookback_days"],
        step=1,
        key="recent_signal_lookback_days",
        help="How many days back to consider a bull-flag entry 'recent'.",
    )
    show_only_recent = st.sidebar.checkbox(
        "Only show tickers with a recent bull-flag signal",
        value=True,
    )

    with st.sidebar.expander("Confidence Filter (optional)", expanded=False):
        min_confidence_cagr = st.slider(
            "Min historical CAGR (strategy %, optional)",
            min_value=-20.0,
            max_value=60.0,
            value=st.session_state["min_confidence_cagr"],
            step=1.0,
            key="min_confidence_cagr",
            help="Drop names with weak historical CAGR under this threshold.",
        )
        min_trades = st.slider(
            "Min # trades (optional)",
            min_value=0,
            max_value=200,
            value=st.session_state["min_trades"],
            step=1,
            key="min_trades",
            help="Avoid names with too few historical signals.",
        )

    run_button = st.sidebar.button("Run Multi-Ticker Scan")

    if not run_button:
        st.info(
            "Choose a universe (manual or index), adjust settings, and click "
            "**Run Multi-Ticker Scan**."
        )
        st.caption(
            "Zaheer defaults (strict)\n"
            "This preset tries to mimic the way Zaheer describes his process in the article: only trade powerful "
            "leaders in strong uptrends, after a clean pause, with tight risk control.\n\n"
            "**Breakout – “New high after a long climb”**  \n"
            "We only consider a stock if it’s breaking out to about a 6-month high, and today’s close is at least "
            "0.5% above the highest close in that window.  \n"
            "Idea: we only want stocks that are pushing into fresh high ground, not stuck in old ranges.\n\n"
            "**Prior trend – “It’s already been a monster”**  \n"
            "Before that breakout, the stock must have been in an uptrend for at least 18 months, and gained at "
            "least 120% over that time (more than doubling).  \n"
            "Idea: we only trade proven leaders that have already shown big, persistent strength.\n\n"
            "**Consolidation – “A healthy rest, not a crash”**  \n"
            "After the strong trend, price must have gone through a sideways or gently down pause for at least 3 "
            "months, and that range can’t be more than 40% tall from low to high.  \n"
            "Idea: we look for orderly bases, not wild, deep corrections.\n\n"
            "**Bull flags – “Short pause just before launch”**  \n"
            "After the breakout, we look for small “flags” that last 4–25 trading days (about 1–5 weeks). During this "
            "flag, price is allowed to pull back up to 15% from the breakout close, but no more.  \n"
            "Idea: we enter when the stock reclaims the breakout level after a shallow, controlled pullback.\n\n"
            "**Risk management – “Small bet, let winners grow”**  \n"
            "On each trade we risk 1% of total account equity from entry down to the stop loss. As the trade works, "
            "we trail a stop 15% below the highest closing price since entry.  \n"
            "Idea: if the stock runs, we stay in and let it compound; if it fails, we lose small and move on."
        )
        return

    # ----- Build ticker list -----
    if universe_mode == "Manual list":
        raw = tickers_text.replace("\n", " ").replace(",", " ")
        tickers = [t.strip().upper() for t in raw.split() if t.strip()]
    else:
        tickers = get_index_universe(index_choice)

    if not tickers:
        if universe_mode == "Manual list":
            st.error("No valid tickers provided.")
        else:
            st.error(f"No tickers returned for index '{index_choice}'.")
        return

    st.sidebar.success(f"Scanning {len(tickers)} tickers.", icon="📈")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    rows = []
    errors = []

    progress = st.progress(0, text="Running model across tickers...")
    for idx, t in enumerate(tickers, start=1):
        try:
            stats = run_model_on_ticker(
                ticker=t,
                start_date=start_str,
                end_date=end_str,
                breakout_lookback=breakout_lookback,
                breakout_buffer_pct=breakout_buffer_pct,
                max_flag_pullback_pct=max_flag_pullback_pct,
                min_flag_len=min_flag_len,
                max_flag_len=max_flag_len,
                min_trend_months=min_trend_months,
                min_trend_return_pct=min_trend_return_pct,
                min_consol_months=min_consol_months,
                max_consol_height_pct=max_consol_height_pct,
                starting_equity=starting_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                trailing_stop_pct=trailing_stop_pct,
                recent_signal_lookback_days=recent_signal_lookback_days,
            )
            if stats is None:
                errors.append(f"{t}: no data")
                continue

            rows.append(
                {
                    "Ticker": t,
                    "Recent": "✅" if stats["has_recent_signal"] else "—",
                    "Has Recent Signal": stats["has_recent_signal"],
                    "Last Signal Date": stats["last_signal_date"],
                    "Num Trades": stats["num_trades"],
                    "P/L $": stats.get("final_equity", starting_equity) - starting_equity,
                    "CAGR %": stats["cagr_pct"],
                    "Total Return %": stats["total_return_pct"],
                    "Max DD %": stats["max_drawdown_pct"],
                    "Win Rate %": stats["win_rate"],
                    "Avg Trade %": stats["avg_trade_return"],
                }
            )
        except Exception as e:
            errors.append(f"{t}: {e}")
        finally:
            progress.progress(idx / len(tickers), text=f"Scanned {idx}/{len(tickers)}")
    progress.empty()

    if errors:
        st.warning(
            "Some tickers could not be processed:\n\n"
            + "\n".join(f"- {msg}" for msg in errors)
        )

    if not rows:
        st.error("No results produced. Check tickers, index, or date range.")
        return

    def _matches_preset(preset: dict) -> bool:
        return all(st.session_state.get(k, preset[k]) == preset[k] for k in preset)

    if _matches_preset(ZAHEER_DEFAULT):
        preset_desc = "Zaheer Tight: Only very strong leaders with long trends and clean, shallow flags."
    elif _matches_preset(LOOSER_PRESET):
        preset_desc = "Looser / Exploratory: More forgiving version to surface more potential ideas."
    else:
        preset_desc = "Custom settings based on current sliders."

    results_df = pd.DataFrame(rows)

    # Confidence = strategy CAGR for now (optional ranking)
    results_df["Confidence"] = results_df["CAGR %"]
    results_df["Has Recent Signal"] = results_df["Has Recent Signal"].fillna(False)
    results_df["Num Trades"] = results_df["Num Trades"].fillna(0).astype(int)
    results_df["Confidence"] = results_df["Confidence"].fillna(-np.inf)

    # Normalize last signal date to datetime (NaT allowed) for stable sorting
    if "Last Signal Date" in results_df.columns:
        results_df["Last Signal Date"] = pd.to_datetime(
            results_df["Last Signal Date"], errors="coerce"
        )

    # --- Apply filters ---
    mask = pd.Series(True, index=results_df.index)

    if show_only_recent:
        mask &= results_df["Has Recent Signal"]

    mask &= results_df["Confidence"] >= min_confidence_cagr
    mask &= results_df["Num Trades"] >= min_trades

    filtered_df = results_df[mask].copy()

    # Sort: recent signals first, then by last signal date, then by Confidence
    sort_cols = ["Has Recent Signal", "Last Signal Date", "Confidence"]
    if not filtered_df.empty:
        filtered_df.sort_values(
            by=sort_cols,
            ascending=[False, False, False],
            inplace=True,
        )
    results_df.sort_values(
        by=sort_cols,
        ascending=[False, False, False],
        inplace=True,
    )

    # ----- Display filtered results -----
    st.subheader("Scan Results (Filtered)")
    st.caption(f"Preset description: {preset_desc}")
    msg_parts = []
    if show_only_recent:
        msg_parts.append(
            f"tickers with a bull-flag entry in the last {recent_signal_lookback_days} days"
        )
    else:
        msg_parts.append("all tickers (recent signal not required)")
    msg_parts.append(f"strategy CAGR ≥ {min_confidence_cagr:.1f}% (optional filter)")
    if min_trades > 0:
        msg_parts.append(f"at least {min_trades} trades")

    st.markdown(
        "Filtered by: **" + " and ".join(msg_parts) + "** over the selected period."
    )

    if filtered_df.empty:
        st.warning(
            "No tickers passed the current filters. Try lowering trend return %, "
            "loosening consolidation height, or unchecking 'Only show recent'."
        )
    else:
        filtered_df_display = filtered_df.copy()
        filtered_df_display["Last Signal Date"] = (
            filtered_df_display["Last Signal Date"].dt.date
        )
        st.dataframe(
            filtered_df_display.style.format(
                {
                    "P/L $": "{:,.0f}",
                    "CAGR %": "{:.1f}",
                    "Total Return %": "{:.1f}",
                    "Max DD %": "{:.1f}",
                    "Win Rate %": "{:.1f}",
                    "Avg Trade %": "{:.1f}",
                    "Total Return %": "{:.1f}",
                    "Confidence": "{:.1f}",
                }
            )
        )

    # ----- Raw results -----
    st.subheader("Raw Results (All Tickers)")
    st.caption(f"Preset description: {preset_desc}")
    st.dataframe(
        results_df.style.format(
            {
                "P/L $": "{:,.0f}",
                "CAGR %": "{:.1f}",
                "Total Return %": "{:.1f}",
                "Max DD %": "{:.1f}",
                "Win Rate %": "{:.1f}",
                "Avg Trade %": "{:.1f}",
                "Confidence": "{:.1f}",
            }
        )
    )

    st.caption(
        "Column definitions:  \n"
        "• **Ticker**: symbol being scanned.  \n"
        "• **Recent**: quick tag (✅/—) if a bull-flag entry occurred in the recent-signal window.  \n"
        "• **Has Recent Signal**: boolean form of the recent tag (used for sorting/filtering).  \n"
        "• **Last Signal Date**: most recent bull-flag entry date (if any).  \n"
        "• **Num Trades**: count of historical strategy trades over the period.  \n"
        "• **P/L $**: dollar profit/loss from the starting equity to final equity.  \n"
        "• **CAGR %**: strategy compound annual growth rate.  \n"
        "• **Total Return %**: strategy total return over the period.  \n"
        "• **Max DD %**: strategy max drawdown.  \n"
        "• **Win Rate %**: percent of trades with positive return.  \n"
        "• **Avg Trade %**: average trade return.  \n"
        "• **Confidence**: currently set to the strategy CAGR %, used for ranking / optional filtering.  \n"
        "Use this scanner to **find current Zaheer-style setups**, then inspect individual tickers "
        "in the single-ticker app for exact entries, stops, and equity curves."
    )


if __name__ == "__main__":
    main()
