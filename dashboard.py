import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

from main import (
    MODEL_FEATURES,
    PROJECT_FOLDER,
    STOCKS,
    day10_result_for_stock,
    load_stock_csv,
    make_model_table,
    time_train_test_split,
)

FEATURE_LABELS = {
    "Previous_Close": "Previous Close",
    "Close_Pct_Change": "Close % Change",
    "Volume_Pct_Change": "Volume % Change",
    "Close_5Day_MA": "5-Day Moving Average",
}

CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def flatten_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]).strip() for col in data.columns]
    return data


def latest_traded_price(ticker):
    """Last traded price for today, even if the regular Close is still blank."""
    live = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if live is not None and not live.empty:
        live = flatten_columns(live)
        closes = live["Close"].dropna() if "Close" in live.columns else pd.Series(dtype=float)
        if not closes.empty:
            return float(closes.iloc[-1])

    ticker_obj = yf.Ticker(ticker)
    last_price = getattr(ticker_obj.fast_info, "last_price", None)
    if last_price is not None and pd.notna(last_price):
        return float(last_price)
    return None


def fill_blank_prices(data, ticker):
    """If today's Close is blank (market still open), use the last traded price."""
    blank_close = data["Close"].isna()
    blank_adj = data["Adj Close"].isna()
    if not (blank_close.any() or blank_adj.any()):
        return data

    price = latest_traded_price(ticker)
    if price is None:
        fallback = data[["Open", "High", "Low"]].mean(axis=1)
        data.loc[blank_close, "Close"] = fallback[blank_close]
        data.loc[blank_adj, "Adj Close"] = data.loc[blank_adj, "Close"]
        return data

    data.loc[blank_close, "Close"] = price
    data.loc[blank_adj, "Adj Close"] = price
    return data


def update_all_stock_csvs():
    """Refresh the six CSVs. Dates after today are dropped. Blank Closes are filled."""
    today = pd.Timestamp.now().normalize()
    newest_dates = {}

    for stock in STOCKS:
        data = yf.download(
            stock["ticker"],
            period="5y",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            raise RuntimeError(f"No rows downloaded for {stock['ticker']}.")

        data = flatten_columns(data).reset_index()
        if "Date" not in data.columns and "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "Date"})

        missing = [name for name in CSV_COLUMNS if name not in data.columns]
        if missing:
            raise RuntimeError(f"{stock['ticker']} download missing columns: {missing}")

        data = data[CSV_COLUMNS]
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data = data.sort_values("Date")
        data = data[data["Date"] <= today]
        data = fill_blank_prices(data, stock["ticker"])
        data = data.dropna(subset=CSV_COLUMNS)
        if data.empty:
            raise RuntimeError(f"{stock['ticker']} had no usable market days.")

        data.to_csv(PROJECT_FOLDER / stock["csv"], index=False)
        newest_dates[stock["ticker"]] = data["Date"].max().date()

    return newest_dates


def draw_price_chart(prices):
    """Dark navy close chart with a white/light-gray price line and a faint fill."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.patch.set_facecolor("#122844")
    ax.set_facecolor("#122844")
    ax.fill_between(prices["Date"], prices["Close"], color="#F4F7FA", alpha=0.10)
    ax.plot(prices["Date"], prices["Close"], color="#F4F7FA", linewidth=2.1)
    ax.grid(True, color="#C8D0D8", alpha=0.18, linewidth=0.6)
    ax.tick_params(colors="#C8D0D8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2b4d78")
    ax.spines["bottom"].set_color("#2b4d78")
    ax.set_ylabel("Close ($)", color="#C8D0D8")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def prediction_history_table(stock):
    """
    Last 10 test days that already have a next-day actual.
    Same older-train / later-test split as Historical Test Accuracy.
    Newest CSV day is left out because its next Close is unknown.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labeled = make_model_table(stock)
    train, test = time_train_test_split(labeled)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[MODEL_FEATURES], train["Target"])
    guessed = model.predict(test[MODEL_FEATURES])

    history = pd.DataFrame(
        {
            "Date": pd.to_datetime(test["Date"]).dt.date,
            "Prediction": ["UP" if code == 1 else "DOWN" for code in guessed],
            "Actual Result": [
                "UP" if target == 1 else "DOWN" for target in test["Target"]
            ],
        }
    )
    history["Correct?"] = [
        "Yes" if prediction == actual else "No"
        for prediction, actual in zip(history["Prediction"], history["Actual Result"])
    ]
    return history.tail(10).iloc[::-1].reset_index(drop=True)


def info_card(label, value, extra_class=""):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">{label}</div>
            <div class="card-value {extra_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Stock Prediction Agent", layout="wide")
st.markdown(
    """
    <style>
      .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #071428;
        color: #e8eef7;
      }
      .stApp { padding-top: 1rem; }
      h1, h2, h3, p, label, li, .stMarkdown { color: #ffffff !important; }
      [data-testid="stCaption"] { color: #C8D0D8 !important; }
      .stButton > button {
        background-color: #1d4f9c;
        color: #ffffff;
        border: 1px solid #2b4d78;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        font-size: 1rem;
      }
      .stButton > button:hover { background-color: #2a63bd; color: #ffffff; }
      div[data-testid="stSelectbox"] {
        background: #122844;
        border: 1px solid #2b4d78;
        border-radius: 10px;
        padding: 0.35rem 0.5rem 0.55rem 0.5rem;
      }
      .info-card {
        background: #122844;
        border: 1px solid #2b4d78;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 108px;
        margin-bottom: 0.6rem;
      }
      .card-label {
        color: #C8D0D8;
        font-size: 0.82rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }
      .card-value {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.25;
      }
      .pred-up { color: #3ddc84 !important; }
      .pred-down { color: #ff5c5c !important; }
      [data-testid="stArrowVegaLiteChart"] {
        background: #122844;
        border: 1px solid #2b4d78;
        border-radius: 14px;
        padding: 0.6rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Prediction Agent")

top_left, top_right = st.columns([1, 2], vertical_alignment="bottom")
with top_left:
    if st.button("Update Stock Data"):
        with st.spinner("Downloading the newest stock data..."):
            newest_dates = update_all_stock_csvs()
        newest = max(newest_dates.values())
        st.session_state["update_message"] = (
            f"Updated all six stocks. Newest day: {newest}"
        )
        st.rerun()
    if st.session_state.get("update_message"):
        st.success(st.session_state.pop("update_message"))
with top_right:
    labels = [f"{stock['name']} ({stock['ticker']})" for stock in STOCKS]
    choice = st.selectbox("Select a stock", labels)

stock = STOCKS[labels.index(choice)]
result = day10_result_for_stock(stock)
pred_class = "pred-up" if result["guess"] == "UP" else "pred-down"

c1, c2, c3, c4 = st.columns(4)
with c1:
    info_card("Stock", f"{result['name']} ({result['ticker']})")
with c2:
    info_card("Latest close", f"${result['latest_close']:.2f}")
with c3:
    info_card("Prediction", result["guess"], pred_class)
with c4:
    info_card("Accuracy", f"{result['accuracy']:.1%}")

st.caption(f"Latest data date: {result['latest_date']}")

prices = load_stock_csv(stock).sort_values("Date")
st.subheader("Closing price over time")
draw_price_chart(prices)

feat_col, how_col = st.columns(2)
with feat_col:
    st.subheader("Features Used by the Model")
    for feature_name in MODEL_FEATURES:
        st.write(f"- {FEATURE_LABELS.get(feature_name, feature_name)}")
with how_col:
    st.subheader("How the model works")
    st.write("The model uses these four pieces of information from the stock's data.")
    st.write("It uses historical data to learn patterns.")
    st.write("It predicts whether the next trading day will be UP or DOWN.")

st.subheader("Prediction History")
st.dataframe(prediction_history_table(stock), hide_index=True, width="stretch")

st.caption("This project is for educational purposes and is not financial advice.")
