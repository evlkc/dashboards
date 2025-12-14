import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime

from core.data import load_price_data
from core.signals import (
    identify_breakouts,
    compute_strong_trend_mask,
    compute_consolidation_mask,
    identify_bull_flags,
)
from core.backtest import backtest_trades
from core.stats import compute_stats, compute_buy_and_hold_equity


@st.cache_data(show_spinner=False)
def load_price_data_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_price_data(ticker, start, end)


def main():
    st.set_page_config(page_title="Trend / Bull-Flag Model Playground", layout="wide")

    # --- Zaheer default config ---
    ZAHEER_DEFAULT = {
        "ticker": "AAPL",
        "start_date": datetime(2014, 1, 1),
        "end_date": datetime.today(),
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
    }

    # Initialize session state with defaults (only if missing)
    for k, v in ZAHEER_DEFAULT.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ----- Presets -----
    st.sidebar.header("Presets")
    if st.sidebar.button("Apply Zaheer Default"):
        for k, v in ZAHEER_DEFAULT.items():
            st.session_state[k] = v
    st.sidebar.caption("Zaheer default = stricter trend + consolidation settings.")

    # ----- Data controls -----
    st.sidebar.header("Data")
    ticker = st.sidebar.text_input("Ticker", key="ticker")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.sidebar.date_input("Start date", key="start_date")
    with col2:
        end_date = st.sidebar.date_input("End date", key="end_date")

    # ----- Investment (was Risk Management) -----
    with st.sidebar.expander("Investment", expanded=False):
        starting_equity = st.number_input(
            "Starting equity ($)",
            min_value=1000.0,
            max_value=1_000_000.0,
            step=1000.0,
            key="starting_equity",
            help="Equity used for sizing and equity curve.",
        )
        risk_per_trade_pct = st.slider(
            "Risk per trade (% of equity)",
            min_value=0.25,
            max_value=5.0,
            step=0.25,
            key="risk_per_trade_pct",
            help="Percent of equity risked per trade based on stop distance.",
        )
        trailing_stop_pct = st.slider(
            "Trailing stop distance (%)",
            min_value=2.0,
            max_value=40.0,
            step=1.0,
            key="trailing_stop_pct",
            help="Stop follows highest close by this %. Lower = tighter.",
        )

    # ----- Breakout & bull-flag -----
    st.sidebar.header("Breakout & Bull Flag Settings")
    c_breakout, c_buffer = st.sidebar.columns(2)
    with c_breakout:
        breakout_lookback = st.sidebar.slider(
            "Breakout lookback (days)",
            min_value=20,
            max_value=252,
            step=5,
            key="breakout_lookback",
            help="Lookback window for prior high; 126 ≈ 6 months of trading days.",
        )
    with c_buffer:
        breakout_buffer_pct = st.sidebar.slider(
            "Breakout buffer (%)",
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            key="breakout_buffer_pct",
            help="Extra % above the prior high to confirm a breakout.",
        )
    c_pullback, c_flaglen = st.sidebar.columns(2)
    with c_pullback:
        max_flag_pullback_pct = st.sidebar.slider(
            "Max flag pullback (%)",
            min_value=2.0,
            max_value=30.0,
            step=1.0,
            key="max_flag_pullback_pct",
            help="How deep the flag can retrace from the breakout close.",
        )
    with c_flaglen:
        min_flag_len = st.sidebar.slider(
            "Min flag length (days)",
            min_value=3,
            max_value=30,
            step=1,
            key="min_flag_len",
            help="Shortest acceptable flag/base length.",
        )
    max_flag_len = st.sidebar.slider(
        "Max flag length (days)",
        min_value=5,
        max_value=60,
        step=1,
        key="max_flag_len",
        help="Longest acceptable flag/base length.",
    )

    # ----- Prior trend filter -----
    st.sidebar.header("Prior Trend Filter")
    c_trend_len, c_trend_ret = st.sidebar.columns(2)
    with c_trend_len:
        min_trend_months = st.sidebar.slider(
            "Min trend length (months)",
            min_value=6,
            max_value=36,
            step=1,
            key="min_trend_months",
            help="Require at least this many months of prior uptrend.",
        )
    with c_trend_ret:
        min_trend_return_pct = st.sidebar.slider(
            "Min trend return (%)",
            min_value=20.0,
            max_value=300.0,
            step=5.0,
            key="min_trend_return_pct",
            help="Minimum % gain over the trend window.",
        )
    require_trend = st.sidebar.checkbox(
        "Require strong prior trend",
        value=True,
        key="require_trend",
    )

    # ----- Consolidation filter -----
    st.sidebar.header("Consolidation Filter")
    c_consol_len, c_consol_ht = st.sidebar.columns(2)
    with c_consol_len:
        min_consol_months = st.sidebar.slider(
            "Min consolidation (months)",
            min_value=2,
            max_value=24,
            step=1,
            key="min_consol_months",
            help="Minimum duration of the base/range.",
        )
    with c_consol_ht:
        max_consol_height_pct = st.sidebar.slider(
            "Max consolidation height (%)",
            min_value=5.0,
            max_value=60.0,
            step=1.0,
            key="max_consol_height_pct",
            help="Max % top-to-bottom height of the range.",
        )

    # ----- Risk management -----

    run_button = st.sidebar.button("Run Backtest")

    if not run_button:
        st.info("Set your parameters in the sidebar and click **Run Backtest**.")
        return

    # =========================
    # Load data
    # =========================
    with st.spinner("Loading data..."):
        df = load_price_data_cached(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if df.empty:
        st.error("No data loaded. Check ticker or date range.")
        return

    st.subheader(f"Price Data: {ticker}")
    price_df = df.reset_index().rename(columns={"index": "date"})
    price_chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Close:Q", title="Close"),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Close:Q", title="Close", format=".2f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(price_chart, use_container_width=True)

    # =========================
    # Signals: breakouts, trend, consolidation, bull flags
    # =========================
    with st.spinner("Finding breakouts, trend filters, consolidations, and bull flags..."):
        breakout_flags = identify_breakouts(df, breakout_lookback, breakout_buffer_pct)
        strong_trend_mask = compute_strong_trend_mask(
            df, min_trend_months, min_trend_return_pct
        )
        consolidation_mask = compute_consolidation_mask(
            df, min_consol_months, max_consol_height_pct
        )

        if require_trend:
            filtered_breakouts = breakout_flags & strong_trend_mask & consolidation_mask
        else:
            filtered_breakouts = breakout_flags & consolidation_mask

        trades = identify_bull_flags(
            df,
            filtered_breakouts,
            max_flag_pullback_pct,
            min_flag_len,
            max_flag_len,
        )

    total_breakouts = int(breakout_flags.sum())
    total_after_trend = int((breakout_flags & strong_trend_mask).sum())
    total_after_consol = int(filtered_breakouts.sum())

    st.write(
        f"Raw breakout signals: **{total_breakouts}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"After prior trend filter: **{total_after_trend}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"After trend + consolidation: **{total_after_consol}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Bull-flag entries found: **{len(trades)}**"
    )

    # =========================
    # Backtest + benchmark
    # =========================
    with st.spinner("Running backtest and benchmark..."):
        equity_curve, trade_results = backtest_trades(
            df,
            trades,
            starting_equity,
            risk_per_trade_pct,
            trailing_stop_pct,
        )

        stats = compute_stats(equity_curve, trade_results, starting_equity)

        # Buy & hold benchmark
        bh_equity_curve = compute_buy_and_hold_equity(df, starting_equity)
        bh_stats = compute_stats(bh_equity_curve, [], starting_equity)

        # Make it explicit when no trades actually fired
        if len(trade_results) == 0:
            st.warning(
                "No trades were taken with these settings on this ticker. "
                "The strategy equity stays flat at the starting equity; "
                "only the buy-and-hold curve moves."
            )

    # =========================
    # Summary metrics
    # =========================
    st.subheader("Backtest Summary vs Buy & Hold")

    if stats and bh_stats:
        # Strategy metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Total Return (%)", f"{stats['total_return_pct']:.1f}")
        c2.metric(
            "Strategy CAGR (%)",
            f"{stats['cagr_pct']:.1f}"
            if not np.isnan(stats["cagr_pct"])
            else "N/A",
        )
        c3.metric("Strategy Max Drawdown (%)", f"{stats['max_drawdown_pct']:.1f}")

        # Benchmark metrics
        d1, d2, d3 = st.columns(3)
        d1.metric("Buy & Hold Total Return (%)", f"{bh_stats['total_return_pct']:.1f}")
        d2.metric(
            "Buy & Hold CAGR (%)",
            f"{bh_stats['cagr_pct']:.1f}"
            if not np.isnan(bh_stats["cagr_pct"])
            else "N/A",
        )
        d3.metric(
            "Buy & Hold Max Drawdown (%)",
            f"{bh_stats['max_drawdown_pct']:.1f}"
        )

        # Dollar P/L
        e_pl1, e_pl2 = st.columns(2)
        strat_pl = stats["final_equity"] - stats["starting_equity"]
        bh_pl = bh_stats["final_equity"] - bh_stats["starting_equity"]
        e_pl1.metric("Strategy P/L ($)", f"{strat_pl:,.0f}")
        e_pl2.metric("Buy & Hold P/L ($)", f"{bh_pl:,.0f}")

        # Outperformance
        strat_tr = stats["total_return_pct"]
        bh_tr = bh_stats["total_return_pct"]
        rel_outperformance = strat_tr - bh_tr  # percentage points

        e1, e2, e3 = st.columns(3)
        e1.metric("# Trades", stats["num_trades"])
        e2.metric(
            "Win Rate (%)",
            f"{stats['win_rate']:.1f}"
            if not np.isnan(stats["win_rate"])
            else "N/A",
        )
        e3.metric(
            "Avg Trade Return (%)",
            f"{stats['avg_trade_return']:.1f}"
            if not np.isnan(stats["avg_trade_return"])
            else "N/A",
        )

        st.markdown(
            f"**Strategy vs Buy & Hold:** Strategy outperformance over the period: "
            f"**{rel_outperformance:+.1f} percentage points** total return."
        )

        st.markdown(
            "**What is CAGR?**  \n"
            "CAGR stands for *Compound Annual Growth Rate*. It is the single, constant yearly rate "
            "that would turn your starting equity into your final equity over the entire period, "
            "assuming all profits are reinvested. It smooths out the year-to-year ups and downs so "
            "you can compare different strategies or tickers on an apples-to-apples “% per year” basis."
        )

    # -----------------------------
    # Equity curves
    # -----------------------------
    st.subheader("Equity Curve: Strategy vs Buy & Hold")

    def _sanitize_curve(curve, name, index, fallback_value):
        """Return a 1-D Series on the provided index with NaNs forward-filled."""
        if isinstance(curve, pd.Series):
            s = curve.copy()
        elif isinstance(curve, pd.DataFrame):
            s = curve.iloc[:, 0].copy()
        else:
            s = pd.Series(np.asarray(curve).ravel())

        s.name = name
        s = s.reindex(index)

        if s.isna().all():
            s = pd.Series(fallback_value, index=index, name=name)
        else:
            if pd.isna(s.iloc[0]) and fallback_value is not None:
                s.iloc[0] = fallback_value
            s = s.ffill()

        return s

    strategy_curve = _sanitize_curve(
        equity_curve, "Strategy", df.index, fallback_value=starting_equity
    )
    bh_curve = _sanitize_curve(
        bh_equity_curve, "Buy & Hold", df.index, fallback_value=starting_equity
    )

    # Normalize each curve to 1.0 at the start so you can see relative growth
    norm_df = pd.DataFrame(
        {
            "Strategy": strategy_curve / strategy_curve.iloc[0],
            "Buy & Hold": bh_curve / bh_curve.iloc[0],
        },
        index=df.index,
    )

    norm_df = norm_df.copy()
    norm_df["date"] = df.index
    norm_long = norm_df.reset_index(drop=True).melt(
        "date", var_name="Series", value_name="Growth"
    )

    norm_chart = (
        alt.Chart(norm_long)
        .mark_line()
        .encode(
            x="date:T",
            y=alt.Y("Growth:Q", scale=alt.Scale(type="log"), title="Growth (x, log)"),
            color="Series:N",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Growth:Q", title="Growth (x)", format=".2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    st.altair_chart(norm_chart, use_container_width=True)

    st.caption(
        "Normalized to 1.0 on the first bar; log scale keeps the strategy visible even if buy & hold compounds faster."
    )

    # ---- Strategy-only view in % terms ----
    st.subheader("Strategy Return Since Start (%)")

    strategy_return_pct = (strategy_curve / strategy_curve.iloc[0] - 1.0) * 100.0
    strategy_return_df = strategy_return_pct.reset_index()
    strategy_return_df.columns = ["date", "return_pct"]
    strategy_return_chart = (
        alt.Chart(strategy_return_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("return_pct:Q", title="Return since start (%)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("return_pct:Q", title="Return (%)", format=".2f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(strategy_return_chart, use_container_width=True)

    st.caption(
        "Strategy-only curve, shown as % return since the first bar. "
        "If this line is moving while buy & hold is huge, your signals are working "
        "even if the absolute dollars are small."
    )

    # =========================
    # Bull-flag entries + completed trades
    # =========================
    st.subheader("Bull-Flag Entry Signals")

    entries_df = pd.DataFrame(trades)
    if not entries_df.empty:
        # Drop signals that never produced a valid entry price
        entries_df = entries_df[entries_df["entry_price"].notna()]

        if entries_df.empty:
            st.write(
                "Bull-flag patterns were detected, but none produced a valid entry "
                "price with the current settings."
            )
        else:
            entries_df = entries_df[
                [
                    "breakout_date",
                    "breakout_price",
                    "entry_date",
                    "entry_price",
                    "stop_price",
                ]
            ]

            # Convert timestamps to plain dates for display
            entries_df["breakout_date"] = pd.to_datetime(
                entries_df["breakout_date"]
            ).dt.date
            entries_df["entry_date"] = pd.to_datetime(
                entries_df["entry_date"]
            ).dt.date

            # Initial risk in %
            entries_df["risk_%"] = (
                (entries_df["entry_price"] - entries_df["stop_price"])
                / entries_df["entry_price"]
                * 100.0
            )

            st.dataframe(
                entries_df.style.format(
                    {
                        "breakout_price": "{:,.2f}",
                        "entry_price": "{:,.2f}",
                        "stop_price": "{:,.2f}",
                        "risk_%": "{:.1f}",
                    }
                )
            )
    else:
        st.write("No bull-flag signals with the current settings.")

    st.subheader("Completed Trades (Backtest)")

    trades_df = pd.DataFrame(trade_results)
    if not trades_df.empty:
        trades_df = trades_df[
            [
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "shares",
                "pnl",
                "return_pct",
            ]
        ]

        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.date
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.date

        # Guard against NaNs so we don't see "None" everywhere
        trades_df["shares"] = trades_df["shares"].fillna(0.0)
        trades_df["pnl"] = trades_df["pnl"].fillna(0.0)
        trades_df["return_pct"] = trades_df["return_pct"].fillna(0.0)

        st.dataframe(
            trades_df.style.format(
                {
                    "entry_price": "{:,.2f}",
                    "exit_price": "{:,.2f}",
                    "shares": "{:,.2f}",
                    "pnl": "{:,.2f}",
                    "return_pct": "{:.2f}",
                }
            )
        )
    else:
        st.write(
            "No completed trades (strategy may still be in an open position or "
            "found no entries)."
        )

    st.caption(
        "This is a simplified but modular stub: core logic lives in the `core/` package, "
        "while this file focuses on the Streamlit UI. "
        "You can now reuse the engine for other apps (CLI, API, multi-ticker scanner, etc.)."
    )


if __name__ == "__main__":
    main()
