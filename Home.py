import streamlit as st

st.set_page_config(page_title="Bull-Flag Toolkit", layout="wide")

st.title("I’ve Averaged 30% A Year In The Stock Market For A Decade With This One Simple Change")
st.markdown(
    "Use the sidebar page selector to switch between the single-ticker playground and "
    "the multi-ticker scanner. Both use the same Zaheer-style breakout + trend + consolidation engine."
)

st.subheader("Pages")
st.markdown(
    "- **Single Ticker**: Apply the Zaheer-style model to one symbol. It checks for breakouts to new highs, "
    "requires a strong prior uptrend, validates a shallow consolidation (bull flag/base), then backtests entries "
    "with sizing and trailing stops. You can inspect signals, trade list, and equity curves vs buy-and-hold.\n"
    "- **Scanner**: Run the same model across an index (S&P 500, Dow 30, Nasdaq 100) or a custom list. It surfaces "
    "tickers with recent bull-flag entries, plus quick stats (CAGR, trades, drawdown, win rate) for context. Use this "
    "to find current setups, then jump into Single Ticker for deeper inspection."
)

st.info("Tip: start with the Scanner to find candidates, then open Single Ticker to inspect entries.")

st.markdown(
    "**We don’t fight the long-term uptrend.**  \n"
    "> We build models to ride it intelligently, and soften the pain when things go wrong."
)

st.subheader("Why this exists")
st.markdown(
    "Most tools either try to predict everything or drown you in indicators.\n\n"
    "SignalPlay takes a different view:\n\n"
    "> **We don’t fight the long-term uptrend.  \n"
    "> We build models to ride it intelligently, and soften the pain when things go wrong.**\n\n"
    "We do that by:\n"
    "- Defining clear, rule-based models (like the Zaheer bull-flag trend follower)\n"
    "- Backtesting them through real crises and recoveries\n"
    "- Showing you exactly how those models behaved vs simple buy-and-hold"
)

st.subheader("Try a quick demo")
st.markdown(
    "1. Open **Single Ticker** in the sidebar.  \n"
    "2. Use ticker `AAPL`, start date `2014-01-01`, keep Zaheer defaults.  \n"
    "3. Click **Run Backtest**.  \n"
    "You'll see how $1,000 would have compounded with the strategy vs buy-and-hold (final equity, CAGR, drawdown, trade list). "
    "Because prices update, the exact dollar result adjusts with the latest data."
)

st.subheader("What this model does")
st.markdown(
    "This toolkit lets you systematically hunt for “Zaheer-style” bull flags – strong stocks making new highs, "
    "pausing in a tight range, then continuing higher – and see how that approach would have behaved in the past.\n\n"
    "Under the hood, both pages use the same engine:\n\n"
    "1) **Find leaders in strong uptrends**  \n"
    "   The model looks for tickers that have already moved a lot (big multi-month uptrend, large % gain). "
    "   The idea is to focus on proven winners, not cheap laggards.\n\n"
    "2) **Require a clean breakout to new highs**  \n"
    "   Price has to break out to roughly a 6-month high, with a small buffer above the old high so we’re not trading random noise.\n\n"
    "3) **Check for a healthy consolidation / base**  \n"
    "   Before the breakout, price must have gone through a sideways or gently down “rest” that isn’t too deep. "
    "   This filters out messy, waterfall-style pullbacks.\n\n"
    "4) **Detect bull-flag entries after the breakout**  \n"
    "   After the breakout, the model looks for short, shallow pullbacks (flags). When price pushes back through the flag, "
    "   that’s your entry signal.\n\n"
    "5) **Apply risk & exit rules**  \n"
    "   Each trade risks only a fixed % of equity (position sizing), and then uses a trailing stop under price to cut losers and "
    "   let winners run. The app tracks equity, drawdowns, and “buy & hold” as a benchmark.\n\n"
    "In short: it’s a trend-following continuation model. It doesn’t predict bottoms; it waits for strength, a pause, and a "
    "continuation pattern, then shows you how that playbook would have performed for a given stock or across an index."
)
