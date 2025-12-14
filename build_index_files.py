import io
import pathlib
import pandas as pd
import requests

# Where the index symbol lists will be written (one ticker per line)
INDEX_DIR = pathlib.Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)


def _clean_tickers(raw):
    """
    Normalize tickers:
    - Uppercase
    - Strip whitespace
    - BRK.B -> BRK-B (Yahoo/ yfinance style)
    - Drop blanks and NaNs
    - Deduplicate and sort
    """
    s = pd.Series(raw, dtype=str).str.upper().str.strip()
    s = s.str.replace(".", "-", regex=False)
    s = s[s != ""]
    s = s.dropna().drop_duplicates()
    return sorted(s.tolist())


def _write_list(tickers, path: pathlib.Path, label: str):
    """Write a list of tickers (one per line) to the given path."""
    tickers = _clean_tickers(tickers)
    with path.open("w") as f:
        for t in tickers:
            f.write(t + "\n")
    print(f"Wrote {len(tickers)} {label} tickers to {path}")


def _fetch_tables(url: str):
    """
    Fetch HTML with a real User-Agent to avoid 403s and parse all tables.

    Returns
    -------
    list[pd.DataFrame]
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; finance-dashboard/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _fetch_csv(url: str):
    """
    Fetch a CSV with a real User-Agent and parse as DataFrame.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; finance-dashboard/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def build_sp500():
    """
    Build S&P 500 symbol list from Wikipedia.

    Writes: indexes/sp500.txt
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _fetch_tables(url)

    df = None
    symbol_cols = ["Symbol", "Ticker symbol", "Ticker"]
    symbol_col = None

    for t in tables:
        for col in symbol_cols:
            if col in t.columns:
                df = t
                symbol_col = col
                break
        if df is not None:
            break

    if df is None or symbol_col is None:
        raise RuntimeError("Could not find a symbol column on the S&P 500 page.")

    tickers = df[symbol_col].tolist()
    _write_list(tickers, INDEX_DIR / "sp500.txt", "S&P 500")


def build_dow30():
    """
    Build Dow 30 symbol list from Wikipedia.

    Writes: indexes/dow30.txt
    """
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = _fetch_tables(url)

    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t
            break

    if df is None:
        raise RuntimeError("Could not find a table with a 'Symbol' column on DJIA page.")

    tickers = df["Symbol"].tolist()
    _write_list(tickers, INDEX_DIR / "dow30.txt", "Dow 30")


def build_nasdaq100():
    """
    Build Nasdaq 100 symbol list from Slickcharts.

    Writes: indexes/nasdaq100.txt
    """
    url = "https://www.slickcharts.com/nasdaq100"
    tables = _fetch_tables(url)
    df = tables[0]  # main table with "Symbol" column
    if "Symbol" not in df.columns:
        raise RuntimeError("Could not find 'Symbol' column on Nasdaq 100 page.")
    tickers = df["Symbol"].tolist()
    _write_list(tickers, INDEX_DIR / "nasdaq100.txt", "Nasdaq-100")


def build_russell1000():
    """
    Build Russell 1000 symbol list with multiple fallbacks:

    1) Slickcharts: https://www.slickcharts.com/russell1000
    2) DataHub CSV: https://datahub.io/core/russell-1000/...
    3) iShares IWB holdings CSV

    Writes: indexes/russell1000.txt
    """
    slick_url = "https://www.slickcharts.com/russell1000"
    datahub_url = "https://datahub.io/core/russell-1000/r/russell-1000.csv"
    ishares_url = (
        "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    )

    def _extract_symbol_col(df, candidates=("Symbol", "Ticker", "Ticker Symbol")):
        for col in candidates:
            if col in df.columns:
                return df[col].tolist()
        raise RuntimeError("Could not find ticker column in fallback data.")

    tickers = None

    # 1) Slickcharts (preferred)
    try:
        tables = _fetch_tables(slick_url)
        df = tables[0]
        tickers = _extract_symbol_col(df)
    except Exception as e:
        print(f"Slickcharts Russell 1000 failed ({e}); trying DataHub CSV...")

    # 2) DataHub CSV (open dataset)
    if tickers is None:
        try:
            df = _fetch_csv(datahub_url)
            tickers = _extract_symbol_col(df)
        except Exception as e:
            print(f"DataHub Russell 1000 failed ({e}); trying iShares IWB holdings CSV...")

    # 3) iShares holdings CSV
    if tickers is None:
        df = _fetch_csv(ishares_url)
        tickers = _extract_symbol_col(df, candidates=("Ticker", "Ticker Symbol", "Symbol"))

    _write_list(tickers, INDEX_DIR / "russell1000.txt", "Russell 1000")


if __name__ == "__main__":
    build_sp500()
    build_dow30()
    build_nasdaq100()
    # build_russell1000()  # Disabled for now
    print("All index files built into ./indexes/")
