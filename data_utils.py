import yfinance as yf
import pandas as pd

def get_asset_data(tickers: list) -> pd.DataFrame:

    data = yf.download(tickers, period="15y", interval="1d")["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    cols_presentes = [t for t in tickers if t in data.columns]
    data = data.loc[:, cols_presentes]

    data = data.dropna()

    return data



def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    data = data.copy()
    train_size = int(len(data) * 0.6)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    return train_data, test_data

def add_overlay(train: pd.DataFrame, test: pd.DataFrame, overlay_size: int = 252):
    overlay = train.iloc[-overlay_size:]
    test = pd.concat([overlay, test])
    test.index = pd.to_datetime(test.index)  # asegura que sean tipo datetime
    return test