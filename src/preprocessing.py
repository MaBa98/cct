import pandas as pd
import numpy as np

def calculate_daily_returns(price_series: pd.Series) -> pd.Series:
    return price_series.pct_change().dropna()

def calculate_log_returns(price_series: pd.Series) -> pd.Series:
    return np.log(price_series / price_series.shift(1)).dropna()

def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def calculate_spread(series1: pd.Series, series2: pd.Series) -> pd.Series:
    return series1 - series2
