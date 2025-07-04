import pandas as pd
from src.preprocessing import (
    calculate_daily_returns,
    rolling_zscore,
    calculate_spread
)


def build_feature_set(price_data: dict, ccteu_tickers: list) -> pd.DataFrame:
    """
    price_data: dict con chiavi = ticker, valori = DataFrame con 'PX_LAST'
    ccteu_tickers: lista dei ticker CCTeu da includere nel modello

    Ritorna: DataFrame con le feature giornaliere per modellistica
    """

    # === DRIVER macro ===
    bund = price_data['RX1 Comdty']['PX_LAST'].rename("bund")
    btp = price_data['IK1 Comdty']['PX_LAST'].rename("btp")
    euribor = price_data['EUR006M Index']['PX_LAST'].rename("euribor")

    # === CCTeu ===
    cct_dfs = []
    for ticker in ccteu_tickers:
        ts = price_data[ticker]['PX_LAST'].rename(f"px_{ticker}")
        cct_dfs.append(ts)

    # === Merge base ===
    df = pd.concat([*cct_dfs, btp, bund, euribor], axis=1)
    
    # === Spread BTP - Bund ===
    df['spread_btp_bund'] = calculate_spread(df['btp'], df['bund'])

    # === Ritorni dei CCTeu ===
    for ticker in ccteu_tickers:
        px_col = f"px_{ticker}"
        ret_col = f"ret_{ticker}"
        df[ret_col] = calculate_daily_returns(df[px_col])

    # === Ritorni driver ===
    df['ret_btp'] = calculate_daily_returns(df['btp'])
    df['ret_bund'] = calculate_daily_returns(df['bund'])
    df['ret_spread_btp_bund'] = calculate_daily_returns(df['spread_btp_bund'])
    df['ret_euribor'] = calculate_daily_returns(df['euribor'])

    # === Z-score (rolling) ===
    df['z_spread_btp_bund'] = rolling_zscore(df['spread_btp_bund'], window=20)
    df['z_euribor'] = rolling_zscore(df['euribor'], window=20)

    # Pulisce i valori mancanti dopo le trasformazioni
    df.dropna(inplace=True)

    return df
