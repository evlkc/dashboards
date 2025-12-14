import numpy as np
import pandas as pd


def backtest_trades(
    df: pd.DataFrame,
    trades: list,
    starting_equity: float,
    risk_per_trade_pct: float,
    trailing_stop_pct: float,
):
    """
    Simple, sequential backtest:

    - One position max.
    - Position size = (risk_per_trade_pct * equity) / (entry_price - stop_price).
    - Trailing stop = highest close since entry * (1 - trailing_stop_pct/100).
    - Mark equity to market every bar.
    """

    df = df.copy()

    # --- Make sure 'Close' is a 1-D float Series ---
    closes = df["Close"]
    # If yfinance gave us a DataFrame or MultiIndex column, squeeze it
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes = pd.Series(closes, index=df.index).astype(float)

    # Pre-map entry_date -> trades to open on that bar (we'll just take the first)
    trades_by_date = {}
    for t in trades:
        dt = pd.to_datetime(t["entry_date"])
        trades_by_date.setdefault(dt, []).append(t)

    equity = float(starting_equity)
    equity_curve = pd.Series(index=df.index, dtype=float)
    trade_results = []

    open_pos = None
    highest_close = None

    for current_date, close in closes.items():
        close = float(close)

        # --- Open new trade if flat and there is a signal today ---
        if open_pos is None and current_date in trades_by_date:
            t = trades_by_date[current_date][0]

            entry_price = float(t["entry_price"])
            stop_price = float(t["stop_price"])
            risk_per_share = entry_price - stop_price

            if risk_per_share > 0:
                risk_dollars = equity * (risk_per_trade_pct / 100.0)
                shares = risk_dollars / risk_per_share

                open_pos = {
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "shares": shares,
                }
                highest_close = close

        # --- Manage open trade ---
        if open_pos is not None:
            highest_close = max(highest_close, close)

            trailing_stop = highest_close * (1 - trailing_stop_pct / 100.0)

            # Exit at close if we hit trailing stop
            if close <= trailing_stop:
                exit_price = close
                pnl = (exit_price - open_pos["entry_price"]) * open_pos["shares"]
                equity += pnl

                trade_results.append(
                    {
                        "entry_date": open_pos["entry_date"],
                        "entry_price": open_pos["entry_price"],
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "shares": open_pos["shares"],
                        "pnl": pnl,
                        "return_pct": (exit_price / open_pos["entry_price"] - 1)
                        * 100.0,
                    }
                )

                open_pos = None
                highest_close = None

        # Mark-to-market equity (if a trade is open, equity is “cash + position”)
        if open_pos is not None:
            unrealised = (close - open_pos["entry_price"]) * open_pos["shares"]
            equity_curve.loc[current_date] = equity + unrealised
        else:
            equity_curve.loc[current_date] = equity

    # If still in a trade at the end, close at last price
    if open_pos is not None:
        last_date = df.index[-1]
        last_close = float(closes.iloc[-1])
        pnl = (last_close - open_pos["entry_price"]) * open_pos["shares"]
        equity += pnl

        trade_results.append(
            {
                "entry_date": open_pos["entry_date"],
                "entry_price": open_pos["entry_price"],
                "exit_date": last_date,
                "exit_price": last_close,
                "shares": open_pos["shares"],
                "pnl": pnl,
                "return_pct": (last_close / open_pos["entry_price"] - 1) * 100.0,
            }
        )
        equity_curve.iloc[-1] = equity

    equity_curve = equity_curve.ffill()

    return equity_curve, trade_results
