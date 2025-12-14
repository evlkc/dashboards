# core/stats.py
import numpy as np
import pandas as pd


def compute_stats(equity_curve: pd.Series, trade_results: list, starting_equity: float):
    if equity_curve.empty or equity_curve.dropna().empty:
        return {}

    eq = equity_curve.dropna()
    start_value = float(eq.iloc[0])
    end_value = float(eq.iloc[-1])
    total_return_pct = (end_value / start_value - 1) * 100.0

    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0
    if years > 0 and start_value > 0 and end_value > 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
        cagr_pct = float(np.real(cagr) * 100.0)
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

    # Guard against complex values sneaking through
    if isinstance(cagr_pct, complex):
        cagr_pct = float("nan")
    if isinstance(total_return_pct, complex):
        total_return_pct = float("nan")

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
