# core/data.py
import pandas as pd
import yfinance as yf


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily OHLCV data from Yahoo Finance via yfinance.
    """
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df
