import yfinance as yf
import pandas as pd

def get_asset_data(tickers: list) -> pd.DataFrame:
    """
    Download historical closing price data for given tickers.

    Parameters:
    tickers : list: A list of ticker symbols.

    Returns:
    pd.DataFrame: A DataFrame containing the closing prices of the tickers.
    """

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
    """
    Split the data into training and testing sets (60% train, 40% test).

    Parameters:
    data : pd.DataFrame: The DataFrame to be split.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """

    data = data.copy()
    train_size = int(len(data) * 0.6)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    return train_data, test_data

def add_overlay(train: pd.DataFrame, test: pd.DataFrame, overlay_size: int = 252):
    """

    Add an overlay of the last `overlay_size` rows from the training set to the beginning of the testing set.
    
    Parameters:
    train : pd.DataFrame: The training DataFrame.
    test : pd.DataFrame: The testing DataFrame.
    overlay_size : int: The number of rows to overlay from the training set to the testing set. Default is 252.

    Returns:
    pd.DataFrame: The modified testing DataFrame with the overlay.
    """
    
    overlay = train.iloc[-overlay_size:]
    test = pd.concat([overlay, test])
    test.index = pd.to_datetime(test.index)  # asegura que sean tipo datetime
    return test