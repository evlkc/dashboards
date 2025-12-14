import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


# -----------------------------
# Model helpers
# -----------------------------
def identify_breakouts(df: pd.DataFrame, breakout_lookback: int, breakout_buffer_pct: float) -> pd.Series:
    """
    Breakout = today's close is above the max close of the last N days
    by at least breakout_buffer_pct.
    """
    past_max = df["Close"].rolling(window=breakout_lookback, min_periods=1).max().shift(1)
    buffer = past_max * (1 + breakout_buffer_pct / 100.0)
    breakout = df["Close"] > buffer
    return breakout.fillna(False)


def compute_strong_trend_mask(
    df: pd.DataFrame,
    min_trend_months: int,
    min_trend_return_pct: float,
) -> pd.Series:
    """
    Approximate a 'prior strong trend' by looking back N months
    (N * 21 trading days) and requiring at least min_trend_return_pct gain.
    """
    lookback_days = int(min_trend_months * 21)  # ~trading days per month
    past_close = df["Close"].shift(lookback_days)
    trend_return_pct = (df["Close"] / past_close - 1.0) * 100.0
    mask = trend_return_pct >= min_trend_return_pct
    return mask.fillna(False)


def compute_consolidation_mask(
    df: pd.DataFrame,
    min_consol_months: int,
    max_consol_height_pct: float,
) -> pd.Series:
    """
    Approximate a multi-month consolidation as:

    - Look back N months (~ N * 21 trading days).
    - Compute the range (max high / min low - 1) over that window.
    - Mark True where the range height is <= max_consol_height_pct.

    Result is a boolean Series aligned with df.index, where True means:
    "As of this bar, price has been in a relatively tight range for N months."
    """
    lookback_days = int(min_consol_months * 21)
    rolling_max = df["High"].rolling(window=lookback_days, min_periods=lookback_days).max()
    rolling_min = df["Low"].rolling(window=lookback_days, min_periods=lookback_days).min()

    height_pct = (rolling_max / rolling_min - 1.0) * 100.0
    mask = height_pct <= max_consol_height_pct
    return mask.fillna(False)


def identify_bull_flags(
    df: pd.DataFrame,
    breakout_flags,
    max_flag_pullback_pct: float,
    min_flag_len: int,
    max_flag_len: int,
):
    """
    Very simplified bull-flag logic, using integer indexing:

    - After a breakout day, we watch the next X days (min_flag_len to max_flag_len):
      * Price must NOT pull back more than max_flag_pullback_pct from the breakout close.
      * A valid flag "triggers" when close exceeds the breakout close again.
      * Entry is on the trigger close; stop loss is the lowest low during the flag.
    """

    # Normalize breakout_flags to a 1-D boolean numpy array of length len(df)
    if isinstance(breakout_flags, pd.DataFrame):
        breakout_flags = breakout_flags.iloc[:, 0]
    if isinstance(breakout_flags, pd.Series):
        flags_arr = breakout_flags.to_numpy(dtype=bool)
    else:
        flags_arr = np.asarray(breakout_flags, dtype=bool)

    flags_arr = flags_arr.ravel()
    if flags_arr.shape[0] != len(df):
        if flags_arr.shape[0] > len(df):
            flags_arr = flags_arr[: len(df)]
        else:
            pad = np.zeros(len(df) - flags_arr.shape[0], dtype=bool)
            flags_arr = np.concatenate([flags_arr, pad])

    trades = []
    df = df.copy()

    # Breakout positions as integer indices
    breakout_pos = np.where(flags_arr)[0]

    for b_idx in breakout_pos:
        b_date = df.index[b_idx]
        b_close = float(df["Close"].iloc[b_idx])

        # Window we consider as possible flag
        start_idx = b_idx + 1
        end_idx = min(b_idx + max_flag_len, len(df) - 1)
        if start_idx >= len(df):
            continue

        window = df.iloc[start_idx : end_idx + 1]
        if window.empty:
            continue

        # Check max pullback
        min_low_in_window = float(window["Low"].min())
        pullback_pct = (b_close - min_low_in_window) / b_close * 100.0
        if pullback_pct < 0:
            pullback_pct = 0.0

        if pullback_pct > max_flag_pullback_pct:
            # Pullback too deep, skip this breakout
            continue

        # Require minimum flag length
        if len(window) < min_flag_len:
            continue

        # Trigger when close exceeds breakout close again
        trigger = window[window["Close"] > b_close]
        if trigger.empty:
            continue

        entry_row = trigger.iloc[0]
        entry_date = entry_row.name
        entry_price = float(entry_row["Close"])

        # Stop = lowest low in the whole flag window up to entry
        flag_until_entry = window.loc[:entry_date]
        stop_price = float(flag_until_entry["Low"].min())

        if stop_price >= entry_price:
            continue

        trades.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "breakout_date": b_date,
                "breakout_price": b_close,
            }
        )

    trades = sorted(trades, key=lambda x: x["entry_date"])
    return trades


# -----------------------------
# Backtester (single-ticker)
# -----------------------------
def backtest_trades(
    df: pd.DataFrame,
    trades: list,
    starting_equity: float,
    risk_per_trade_pct: float,
    trailing_stop_pct: float,
):
    """
    Simple, sequential backtest:
    - Only one open position at a time.
    - Position size = (risk_per_trade_pct * equity) / (entry - stop).
    - Trailing stop = highest close since entry * (1 - trailing_stop_pct).
    - Exit when close <= trailing stop.
    """

    equity = float(starting_equity)
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve[:] = np.nan

    trade_results = []
    open_trade = None
    highest_close_since_entry = None

    # Map entry_date -> list of trades (we'll just use the first one to avoid overlap)
    trades_by_date = {}
    for t in trades:
        trades_by_date.setdefault(t["entry_date"], []).append(t)

    for current_date, row in df.iterrows():
        price_close = float(row["Close"])

        # Open a new trade if no open trade and a signal exists today
        if open_trade is None and current_date in trades_by_date:
            t = trades_by_date[current_date][0]
            entry_price = float(t["entry_price"])
            stop_price = float(t["stop_price"])
            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0:
                equity_curve[current_date] = equity
                continue

            risk_amount = equity * (risk_per_trade_pct / 100.0)
            shares = risk_amount / risk_per_share

            open_trade = {
                "entry_date": current_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "shares": shares,
                "risk_amount": risk_amount,
            }
            highest_close_since_entry = price_close

        # Manage open trade
        if open_trade is not None:
            highest_close_since_entry = (
                price_close
                if highest_close_since_entry is None
                else max(float(highest_close_since_entry), price_close)
            )

            trailing_stop = highest_close_since_entry * (1 - trailing_stop_pct / 100.0)
            open_trade["trailing_stop"] = trailing_stop

            if price_close <= trailing_stop:
                exit_price = price_close
                pnl = (exit_price - open_trade["entry_price"]) * open_trade["shares"]
                equity += pnl

                trade_results.append(
                    {
                        "entry_date": open_trade["entry_date"],
                        "entry_price": open_trade["entry_price"],
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "shares": open_trade["shares"],
                        "pnl": pnl,
                        "return_pct": (exit_price / open_trade["entry_price"] - 1)
                        * 100.0,
                    }
                )
                open_trade = None
                highest_close_since_entry = None

        equity_curve[current_date] = equity

    # If a trade remains open at the end, close at last price
    if open_trade is not None:
        last_date = df.index[-1]
        last_price = float(df["Close"].iloc[-1])
        pnl = (last_price - open_trade["entry_price"]) * open_trade["shares"]
        equity += pnl
        trade_results.append(
            {
                "entry_date": open_trade["entry_date"],
                "entry_price": open_trade["entry_price"],
                "exit_date": last_date,
                "exit_price": last_price,
                "shares": open_trade["shares"],
                "pnl": pnl,
                "return_pct": (last_price / open_trade["entry_price"] - 1) * 100.0,
            }
        )
        equity_curve[last_date] = equity

    equity_curve = equity_curve.ffill()
    return equity_curve, trade_results


def compute_stats(equity_curve: pd.Series, trade_results: list, starting_equity: float):
    if equity_curve.empty or equity_curve.dropna().empty:
        return {}

    eq = equity_curve.dropna()
    start_value = float(eq.iloc[0])
    end_value = float(eq.iloc[-1])
    total_return_pct = (end_value / start_value - 1) * 100.0

    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0
    if years > 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
        cagr_pct = cagr * 100.0
    else:
        cagr_pct = float("nan")

    running_max = eq.cummax()
    drawdown = (eq / running_max - 1) * 100.0
    max_drawdown_pct = float(drawdown.min())

    num_trades = len(trade_results)
    if num_trades > 0:
        wins = [t for t in trade_results if t["pnl"] > 0]
        win_rate = len(wins) / num_trades * 100.0
        avg_trade_return = float(np.mean([t["return_pct"] for t in trade_results]))
    else:
        win_rate = float("nan")
        avg_trade_return = float("nan")

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "final_equity": end_value,
        "starting_equity": float(starting_equity),
    }


def compute_buy_and_hold_equity(df: pd.DataFrame, starting_equity: float) -> pd.Series:
    """
    Simple buy-and-hold benchmark:
    - Invest starting_equity at the first close.
    - Let it ride; equity moves in proportion to price.
    """
    if df.empty:
        return pd.Series(dtype=float)
    first_close = float(df["Close"].iloc[0])
    bh_equity = starting_equity * (df["Close"] / first_close)
    bh_equity.name = "Buy & Hold"
    return bh_equity


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Trend / Bull-Flag Model Playground", layout="wide")

    st.title("Trend-Following / Bull-Flag Model (Stub)")
    st.markdown(
        "Experiment with breakout + bull-flag logic, prior trend filters, consolidation filters, "
        "and trailing stops.\n"
        "Compare the strategy to a simple buy-and-hold benchmark on the same ticker and period."
    )

    # Sidebar controls
    st.sidebar.header("Data")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.sidebar.date_input("Start date", value=datetime(2014, 1, 1))
    with col2:
        end_date = st.sidebar.date_input("End date", value=datetime.today())

    st.sidebar.header("Breakout & Bull Flag Settings")
    breakout_lookback = st.sidebar.slider(
        "Breakout lookback (days)", min_value=20, max_value=252, value=126, step=5
    )
    breakout_buffer_pct = st.sidebar.slider(
        "Breakout buffer above prior high (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )
    max_flag_pullback_pct = st.sidebar.slider(
        "Max flag pullback from breakout (%)",
        min_value=2.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
    )
    min_flag_len = st.sidebar.slider(
        "Min flag length (days)", min_value=3, max_value=30, value=5, step=1
    )
    max_flag_len = st.sidebar.slider(
        "Max flag length (days)", min_value=5, max_value=60, value=20, step=1
    )

    st.sidebar.header("Prior Trend Filter")
    min_trend_months = st.sidebar.slider(
        "Min prior trend length (months)", min_value=6, max_value=36, value=18, step=1
    )
    min_trend_return_pct = st.sidebar.slider(
        "Min prior trend return (%)",
        min_value=20.0,
        max_value=300.0,
        value=100.0,
        step=5.0,
    )

    st.sidebar.header("Consolidation Filter")
    min_consol_months = st.sidebar.slider(
        "Min consolidation length (months)",
        min_value=2,
        max_value=24,
        value=6,
        step=1,
    )
    max_consol_height_pct = st.sidebar.slider(
        "Max consolidation height (%)",
        min_value=5.0,
        max_value=60.0,
        value=20.0,
        step=1.0,
    )

    st.sidebar.header("Risk Management")
    starting_equity = st.sidebar.number_input(
        "Starting equity ($)",
        min_value=1000.0,
        max_value=1_000_000.0,
        value=1_000.0,
        step=1000.0,
    )
    risk_per_trade_pct = st.sidebar.slider(
        "Risk per trade (% of equity)",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
    )
    trailing_stop_pct = st.sidebar.slider(
        "Trailing stop distance (%) below highest close",
        min_value=2.0,
        max_value=40.0,
        value=15.0,
        step=1.0,
    )

    run_button = st.sidebar.button("Run Backtest")

    if not run_button:
        st.info("Set your parameters in the sidebar and click **Run Backtest**.")
        return

    # Load data
    with st.spinner("Loading data..."):
        df = load_price_data(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if df.empty:
        st.error("No data loaded. Check ticker or date range.")
        return

    st.subheader(f"Price Data: {ticker}")
    st.line_chart(df["Close"])

    # Identify breakouts + prior trend + consolidation + flags
    with st.spinner("Finding breakouts, trend filters, consolidations, and bull flags..."):
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

    total_breakouts = int(breakout_flags.sum())
    total_after_trend = int((breakout_flags & strong_trend_mask).sum())
    total_after_consol = int(filtered_breakouts.sum())

    st.write(
        f"Raw breakout signals: **{total_breakouts}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"After prior trend filter: **{total_after_trend}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"After trend + consolidation: **{total_after_consol}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Bull-flag entries found: **{len(trades)}**"
    )

    if not trades:
        st.warning(
            "No trades found with the current settings."
            "Strategy equity will stay flat at starting equity, "
            "but the buy-and-hold benchmark is still shown."
            )
        return

    # Backtest
    with st.spinner("Running backtest and benchmark..."):
        if trades:
            equity_curve, trade_results = backtest_trades(
                df,
                trades,
                starting_equity,
                risk_per_trade_pct,
                trailing_stop_pct,
            )
            stats = compute_stats(equity_curve, trade_results, starting_equity)
        else:
            # No trades: flat equity at starting value
            equity_curve = pd.Series(
                data=starting_equity, index=df.index, dtype=float
            )
            trade_results = []
            stats = compute_stats(equity_curve, trade_results, starting_equity)

        # Buy & hold benchmark
        bh_equity_curve = compute_buy_and_hold_equity(df, starting_equity)
        bh_stats = compute_stats(bh_equity_curve, [], starting_equity)

    # Summary metrics
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
        # Equity curves
    st.subheader("Equity Curve: Strategy vs Buy & Hold")

    # Ensure both curves are clean 1-D Series
    def _to_series(curve, name):
        # If it's already a Series, just rename
        if isinstance(curve, pd.Series):
            s = curve.copy()
            s.name = name
            return s
        # If it's a DataFrame, take the first column
        if isinstance(curve, pd.DataFrame):
            s = curve.iloc[:, 0].copy()
            s.name = name
            return s
        # If it's a numpy array or list, flatten it
        arr = np.asarray(curve).ravel()
        return pd.Series(arr, index=df.index, name=name)

    strategy_curve = _to_series(equity_curve, "Strategy")
    bh_curve = _to_series(bh_equity_curve, "Buy & Hold")

    # Align and combine
    equity_df = pd.concat([strategy_curve, bh_curve], axis=1)

    st.line_chart(equity_df)

    # Trade list
    st.subheader("Trade List")
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
        trades_df["entry_date"] = trades_df["entry_date"].dt.date
        trades_df["exit_date"] = trades_df["exit_date"].dt.date
        st.dataframe(
            trades_df.style.format({"pnl": "{:,.2f}", "return_pct": "{:.2f}"})
        )
    else:
        st.write("No completed trades.")

    st.caption(
        "This is still a simplified stub, but now includes prior trend and consolidation filters, "
        "and compares the strategy to a buy-and-hold benchmark on the same ticker and period."
    )


if __name__ == "__main__":
    main()
