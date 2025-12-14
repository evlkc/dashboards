# core/signals.py
import numpy as np
import pandas as pd


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
    (~ N * 21 trading days) and requiring at least min_trend_return_pct gain.
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
    Simplified bull-flag logic, using integer indexing:

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
