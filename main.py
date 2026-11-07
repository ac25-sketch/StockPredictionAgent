"""
Stock Prediction Agent
Day 3: graphs from saved CSVs.
Day 4: extra columns on Apple data for a future model (no AI yet).
Day 5: historical Target (1 = next day Close went up, 0 = it did not).
Day 6: simple classification model that predicts Target from Day 4 features.
Day 7: UP/DOWN guess for the next day after the newest CSV date (no future data).
Day 9: compare Day 6 vs adding Close vs 5-day-MA percent (Day 6/7 functions unchanged).
Day 10: one easy-to-read prediction report for all six stocks.
Does not download any new data from the internet.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save pictures to files (no pop-up window needed).
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent
GRAPHS_FOLDER = PROJECT_FOLDER / "graphs"
PROCESSED_FOLDER = PROJECT_FOLDER / "processed_data"

NEEDED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

STOCKS = [
    {"name": "Apple", "ticker": "AAPL", "csv": "apple_stock_data.csv"},
    {"name": "NVIDIA", "ticker": "NVDA", "csv": "nvidia_stock_data.csv"},
    {"name": "Microsoft", "ticker": "MSFT", "csv": "microsoft_stock_data.csv"},
    {"name": "Tesla", "ticker": "TSLA", "csv": "tesla_stock_data.csv"},
    {"name": "Nike", "ticker": "NKE", "csv": "nike_stock_data.csv"},
    {"name": "Amazon", "ticker": "AMZN", "csv": "amazon_stock_data.csv"},
]

# Day 4 features run for these tickers only. Add more later without rewriting the math.
FEATURE_TICKERS = ["AAPL"]

# Day 6 model inputs. Target and next-day Close are never in this list.
MODEL_FEATURES = [
    "Previous_Close",
    "Close_Pct_Change",
    "Volume_Pct_Change",
    "Close_5Day_MA",
]


def load_stock_csv(stock):
    """Read one company's saved CSV and check it is usable."""
    path = PROJECT_FOLDER / stock["csv"]
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name}. Run the Day 2 download first.")

    table = pd.read_csv(path, parse_dates=["Date"])
    missing = [name for name in NEEDED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    table = table.dropna(subset=NEEDED_COLUMNS)
    if table.empty:
        raise ValueError(f"{path.name} has no rows.")
    if (table["Close"] <= 0).any():
        raise ValueError(f"{path.name} has close prices that are not usable.")

    table = table.sort_values("Date")
    return table


def print_close_stats(stock, table):
    closes = table["Close"]
    print(f"{stock['name']} ({stock['ticker']})")
    print(f"  Days in file: {len(table)}")
    print(f"  Dates: {table['Date'].min().date()} to {table['Date'].max().date()}")
    print(f"  Highest close: ${closes.max():,.2f}")
    print(f"  Lowest close:  ${closes.min():,.2f}")
    print(f"  Average close: ${closes.mean():,.2f}")
    print()


def save_close_graph(stock, table):
    path = GRAPHS_FOLDER / f"{stock['ticker']}_closing_price.png"
    plt.figure(figsize=(10, 5))
    plt.plot(table["Date"], table["Close"], color="tab:blue")
    plt.title(f"{stock['name']} ({stock['ticker']}) closing price over time")
    plt.xlabel("Date")
    plt.ylabel("Close price ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_volume_graph(stock, table):
    path = GRAPHS_FOLDER / f"{stock['ticker']}_trading_volume.png"
    plt.figure(figsize=(10, 5))
    plt.plot(table["Date"], table["Volume"], color="tab:orange")
    plt.title(f"{stock['name']} ({stock['ticker']}) trading volume over time")
    plt.xlabel("Date")
    plt.ylabel("Shares traded")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def add_prediction_features(table):
    """Add helper columns a future model can use. This is not a prediction yet."""
    table = table.sort_values("Date").copy()

    # Previous day's closing price: what Close was on the trading day before this row.
    table["Previous_Close"] = table["Close"].shift(1)

    # Daily % change in close: ((today - yesterday) / yesterday) * 100.
    # Example: yesterday $100, today $105 -> +5.0
    table["Close_Pct_Change"] = table["Close"].pct_change() * 100

    # Daily % change in volume: how much more/less trading happened vs yesterday.
    table["Volume_Pct_Change"] = table["Volume"].pct_change() * 100

    # 5-day moving average of close: mean Close of this day and the 4 days before it.
    table["Close_5Day_MA"] = table["Close"].rolling(window=5).mean()

    return table


def prepare_features_for_stock(stock, table):
    """Save a new CSV with extra columns. The original Day 2 CSV is not changed."""
    featured = add_prediction_features(table)
    out_path = PROJECT_FOLDER / f"{stock['ticker'].lower()}_stock_features.csv"
    featured.to_csv(out_path, index=False)

    preview_columns = [
        "Date",
        "Close",
        "Volume",
        "Previous_Close",
        "Close_Pct_Change",
        "Volume_Pct_Change",
        "Close_5Day_MA",
    ]
    print(f"Day 4 features for {stock['name']} ({stock['ticker']})")
    print(f"Saved {out_path.name} (original {stock['csv']} left unchanged)")
    print("First rows (early days have blanks: there is no 'yesterday' yet,")
    print("and the 5-day average needs 5 days before it can fill in):")
    print(featured[preview_columns].head(8).to_string(index=False))
    print()
    return out_path, featured


def save_close_and_ma_graph(stock, featured):
    """New graph only: Close and 5-day average together. Old graph files stay."""
    path = GRAPHS_FOLDER / f"{stock['ticker']}_close_and_5day_ma.png"
    plt.figure(figsize=(10, 5))
    plt.plot(featured["Date"], featured["Close"], color="tab:blue", label="Closing price")
    plt.plot(
        featured["Date"],
        featured["Close_5Day_MA"],
        color="tab:red",
        label="5-day moving average",
    )
    plt.title(f"{stock['name']} ({stock['ticker']}) close vs 5-day moving average")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved new graph: {path.name}")
    print(f"(Your other graphs in {GRAPHS_FOLDER.name}/ were not deleted.)")
    print()
    return path


def run_graphs():
    print("Stock Prediction Agent — Day 3 graphs")
    print("Using saved CSVs only. Not downloading new data.")
    print()

    GRAPHS_FOLDER.mkdir(exist_ok=True)

    saved = []
    loaded = {}
    for stock in STOCKS:
        try:
            table = load_stock_csv(stock)
            loaded[stock["ticker"]] = (stock, table)
            print("Data check: OK — file has dates, prices, and volume.")
            print_close_stats(stock, table)
            close_path = save_close_graph(stock, table)
            volume_path = save_volume_graph(stock, table)
            saved.append(close_path.name)
            saved.append(volume_path.name)
            print(f"  Saved {close_path.name}")
            print(f"  Saved {volume_path.name}")
            print()
        except Exception as error:
            print(f"Could not graph {stock['name']} ({stock['ticker']}).")
            print(f"Error: {error}")
            print()

    print(f"Graphs folder: {GRAPHS_FOLDER}")
    print(f"Created {len(saved)} graph files.")
    print()
    return loaded


def run_day4_features(loaded):
    print("Day 4 — prepare features for a future model (no AI yet)")
    print()
    for ticker in FEATURE_TICKERS:
        if ticker not in loaded:
            print(f"Skipping {ticker}: CSV was not loaded.")
            continue
        stock, table = loaded[ticker]
        try:
            _, featured = prepare_features_for_stock(stock, table)
            save_close_and_ma_graph(stock, featured)
        except Exception as error:
            print(f"Could not add features for {stock['name']}.")
            print(f"Error: {error}")
            print()


def load_table_for_target(stock):
    """Use Day 4 features when they exist (Apple). Otherwise use the original CSV."""
    features_path = PROJECT_FOLDER / f"{stock['ticker'].lower()}_stock_features.csv"
    if features_path.exists():
        table = pd.read_csv(features_path, parse_dates=["Date"])
        return table.sort_values("Date")
    return load_stock_csv(stock)


def add_historical_target(table):
    """
    Label each day with what the NEXT trading day's Close actually did.
    Target = 1 if tomorrow's Close was higher than today's Close.
    Target = 0 if tomorrow's Close was lower or the same.
    The next Close is used only to make this label, then thrown away
    so it is not an input feature.
    """
    table = table.sort_values("Date").copy()
    next_close = table["Close"].shift(-1)
    table["Target"] = (next_close > table["Close"]).astype(int)
    # Last day has no "tomorrow" in the file, so drop only that row.
    table = table.iloc[:-1]
    return table


def prepare_target_for_stock(stock):
    table = load_table_for_target(stock)
    labeled = add_historical_target(table)

    PROCESSED_FOLDER.mkdir(exist_ok=True)
    out_path = PROCESSED_FOLDER / f"{stock['ticker']}_processed.csv"
    labeled.to_csv(out_path, index=False)

    ones = int((labeled["Target"] == 1).sum())
    zeros = int((labeled["Target"] == 0).sum())

    print(f"{stock['name']} ({stock['ticker']})")
    print(f"Saved {out_path}")
    print(f"Rows kept: {len(labeled)} (dropped the last day; no next Close)")
    print("Example rows (Date, Close, Target):")
    print(labeled[["Date", "Close", "Target"]].head(8).to_string(index=False))
    print(f"Target=1 (next day went up): {ones}")
    print(f"Target=0 (next day down or flat): {zeros}")
    print()
    return out_path


def run_day5_targets():
    print("Stock Prediction Agent — Day 5 historical Target")
    print("Not training a model. Not downloading. Not changing old graphs.")
    print()

    for stock in STOCKS:
        try:
            prepare_target_for_stock(stock)
        except Exception as error:
            print(f"Could not create Target for {stock['name']} ({stock['ticker']}).")
            print(f"Error: {error}")
            print()


def make_model_table(stock):
    """Build Date + Day 4 features + Target in memory. Does not rewrite Day 5 files."""
    table = load_stock_csv(stock)
    table = add_prediction_features(table)
    table = add_historical_target(table)
    needed = MODEL_FEATURES + ["Date", "Close", "Target"]
    table = table[needed].dropna()
    return table.sort_values("Date").reset_index(drop=True)


def time_train_test_split(table, train_ratio=0.8):
    """Older rows train, newer rows test. Dates stay in order (no shuffle)."""
    split_at = int(len(table) * train_ratio)
    train = table.iloc[:split_at]
    test = table.iloc[split_at:]
    return train, test


def train_and_test_one_stock(stock):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    table = make_model_table(stock)
    train, test = time_train_test_split(table)

    X_train = train[MODEL_FEATURES]
    y_train = train["Target"]
    X_test = test[MODEL_FEATURES]
    y_test = test["Target"]

    # Logistic regression: learns a yes/no guess from the four numbers.
    # Scaler: makes a $300 price and a 2% change comparable so one does not drown the other.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    examples = test[["Date", "Close", "Target"]].copy()
    examples["Prediction"] = predictions
    examples["Date"] = pd.to_datetime(examples["Date"]).dt.date

    print(f"{stock['name']} ({stock['ticker']})")
    print(f"Features used: {', '.join(MODEL_FEATURES)}")
    print(f"Train days (older): {len(train)}  {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Test days (later):  {len(test)}  {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"Accuracy on later data: {accuracy:.1%}")
    print("A few later-day examples (Prediction vs actual Target):")
    print(examples.head(8).to_string(index=False))
    print()
    return accuracy


def run_day6_model():
    print("Stock Prediction Agent — Day 6 first machine-learning model")
    print("Predicts Target (1 = next day up, 0 = not up).")
    print("Does not use Target or next-day Close as inputs.")
    print("Does not redo Day 5 files. No website.")
    print()
    print("Features the model is allowed to see:")
    for name in MODEL_FEATURES:
        print(f"  - {name}")
    print()

    for stock in STOCKS:
        try:
            train_and_test_one_stock(stock)
        except Exception as error:
            print(f"Could not train {stock['name']} ({stock['ticker']}).")
            print(f"Error: {error}")
            print()


def predict_up_or_down(stock):
    """
    Guess UP or DOWN for the trading day AFTER the newest day in the CSV.
    Train only on rows that already have a known Target.
    The newest day is not used as a training label (its next Close is unknown).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    featured = add_prediction_features(load_stock_csv(stock)).sort_values("Date")
    latest = featured.iloc[-1]
    latest_date = pd.to_datetime(latest["Date"])

    labeled = add_historical_target(featured)
    labeled = labeled[["Date", "Close", "Target"] + MODEL_FEATURES].dropna()
    train = labeled[pd.to_datetime(labeled["Date"]) < latest_date]

    if train.empty:
        raise RuntimeError("Not enough labeled history to train.")
    if pd.isna(latest[MODEL_FEATURES]).any():
        raise RuntimeError("The newest day is missing Day 4 features.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train[MODEL_FEATURES], train["Target"])

    x_latest = latest[MODEL_FEATURES].to_frame().T
    guess_code = int(model.predict(x_latest)[0])
    classes = list(model.classes_)
    up_chance = float(model.predict_proba(x_latest)[0][classes.index(1)])

    guess = "UP" if guess_code == 1 else "DOWN"
    print(f"{stock['name']} ({stock['ticker']})")
    print(f"Based on close of {latest_date.date()} (${latest['Close']:.2f})")
    print(f"Guess for the next trading day: {guess}")
    print(f"Model chance of UP: {up_chance:.1%}")
    print("Inputs: only Day 4 features from that day and before. No Target. No next Close.")
    print()
    return guess


def run_up_down_predictions():
    print("UP/DOWN prediction system (no future information)")
    print("Newest CSV days are still from the last download, not necessarily today.")
    print()
    for stock in STOCKS:
        try:
            predict_up_or_down(stock)
        except Exception as error:
            print(f"Could not predict {stock['name']} ({stock['ticker']}).")
            print(f"Error: {error}")
            print()


NEW_FEATURE = "Close_vs_5Day_MA_Pct"
FEATURES_WITH_NEW = MODEL_FEATURES + [NEW_FEATURE]


def add_close_vs_ma_pct(table):
    """
    Percent gap between today's Close and the 5-day average.
    Example: average $100, close $103 -> +3.0
    Uses only Close and Close_5Day_MA, both known at today's close.
    """
    table = table.copy()
    table[NEW_FEATURE] = (table["Close"] - table["Close_5Day_MA"]) / table["Close_5Day_MA"] * 100
    return table


def accuracy_with_features(stock, feature_names):
    """Same older-train / later-test split as Day 6. Does not change Day 6 functions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    table = make_model_table(stock)
    if NEW_FEATURE in feature_names:
        table = add_close_vs_ma_pct(table)
        table = table.dropna(subset=feature_names)

    train, test = time_train_test_split(table)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train[feature_names], train["Target"])
    predictions = model.predict(test[feature_names])
    accuracy = accuracy_score(test["Target"], predictions)
    return accuracy, train, test


def describe_change(old_acc, new_acc):
    if new_acc > old_acc + 1e-12:
        return "improved"
    if new_acc < old_acc - 1e-12:
        return "worse"
    return "stayed the same"


def run_new_feature_comparison():
    print("New feature test: Close vs 5-day moving average (%)")
    print("Same train/test dates as Day 6. Day 6 and Day 7 code was not changed.")
    print(f"Added input: {NEW_FEATURE} = (Close - Close_5Day_MA) / Close_5Day_MA * 100")
    print("Not using Target or next-day Close as an input.")
    print()

    print(f"{'Stock':<12} {'Day 6':>8} {'With new feature':>18} {'Change':>8} Result")
    for stock in STOCKS:
        try:
            old_acc, train, test = accuracy_with_features(stock, MODEL_FEATURES)
            new_acc, _, _ = accuracy_with_features(stock, FEATURES_WITH_NEW)
            change_pp = (new_acc - old_acc) * 100
            result = describe_change(old_acc, new_acc)
            print(
                f"{stock['name']:<12} {old_acc:8.1%} {new_acc:18.1%} "
                f"{change_pp:+7.1f}pp {result}"
            )
            print(
                f"  Train {train['Date'].min().date()} to {train['Date'].max().date()} | "
                f"Test {test['Date'].min().date()} to {test['Date'].max().date()}"
            )
        except Exception as error:
            print(f"Could not compare {stock['name']}: {error}")
    print()


def day10_result_for_stock(stock):
    """
    Same Day 6 test accuracy and same Day 7 UP/DOWN rules.
    Does not change those older functions. Does not add features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Historical accuracy: older dates train, later dates test (Day 6).
    labeled = make_model_table(stock)
    train, test = time_train_test_split(labeled)
    tester = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    tester.fit(train[MODEL_FEATURES], train["Target"])
    accuracy = accuracy_score(test["Target"], tester.predict(test[MODEL_FEATURES]))

    # Current guess: newest CSV day only, no Target / no next Close as input (Day 7).
    featured = add_prediction_features(load_stock_csv(stock)).sort_values("Date")
    latest = featured.iloc[-1]
    latest_date = pd.to_datetime(latest["Date"])
    history = add_historical_target(featured)
    history = history[["Date", "Close", "Target"] + MODEL_FEATURES].dropna()
    history = history[pd.to_datetime(history["Date"]) < latest_date]

    predictor = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    predictor.fit(history[MODEL_FEATURES], history["Target"])
    x_latest = latest[MODEL_FEATURES].to_frame().T
    guess_code = int(predictor.predict(x_latest)[0])
    classes = list(predictor.classes_)
    up_chance = float(predictor.predict_proba(x_latest)[0][classes.index(1)])

    return {
        "name": stock["name"],
        "ticker": stock["ticker"],
        "latest_date": latest_date.date(),
        "latest_close": float(latest["Close"]),
        "guess": "UP" if guess_code == 1 else "DOWN",
        "up_chance": up_chance,
        "accuracy": accuracy,
    }


def run_day10_report():
    print("=" * 78)
    print("Stock Prediction Agent — current predictions")
    print("=" * 78)
    print("Question: will the NEXT trading day close higher than the latest Close?")
    print("UP = yes.  DOWN = no (lower or the same).")
    print("Inputs: previous close, daily close % change, daily volume % change,")
    print("        and the 5-day moving average. No Target. No next-day Close.")
    print("Accuracy = how often the model was right on later historical days")
    print("(trained on older days, tested on newer days, dates not shuffled).")
    print("Latest input date should be the last complete day in the CSV: 2026-10-16.")
    print("=" * 78)
    print()
    print(
        f"{'Stock':<12} {'Ticker':<7} {'Latest date':<12} {'Close':>10} "
        f"{'Guess':<6} {'Chance UP':>10} {'Test accuracy':>14}"
    )
    print("-" * 78)

    for stock in STOCKS:
        row = day10_result_for_stock(stock)
        print(
            f"{row['name']:<12} {row['ticker']:<7} {row['latest_date']} "
            f"${row['latest_close']:8.2f}  {row['guess']:<6} "
            f"{row['up_chance']:9.1%} {row['accuracy']:13.1%}"
        )
        print()
        print(f"  {row['name']} ({row['ticker']})")
        print(f"  Latest data date: {row['latest_date']}")
        print(f"  Latest Close: ${row['latest_close']:.2f}")
        print(f"  Prediction for the next trading day: {row['guess']}")
        print(f"  Chance of UP: {row['up_chance']:.1%}")
        print(f"  Historical test accuracy: {row['accuracy']:.1%}")
        print()

    print("-" * 78)
    print("This is a school-project guess from past prices, not financial advice.")
    print("About 50% accuracy is close to a coin flip.")


def main():
    run_day10_report()


if __name__ == "__main__":
    main()
