import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.preprocessing import rolling_zscore

def compute_pca_components(df_prices: pd.DataFrame, n_components=2):
    """
    Applica PCA alla matrice di prezzi dei CCTeu.
    Ritorna: componenti principali, explained variance
    """
    df_clean = df_prices.dropna()
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_clean)
    explained = pca.explained_variance_ratio_
    return pd.DataFrame(components, index=df_clean.index), explained


def compute_cluster_index(df_returns: pd.DataFrame, method='mean'):
    """
    Calcola un indice sintetico di cluster (media o mediana dei rendimenti)
    """
    if method == 'mean':
        return df_returns.mean(axis=1)
    elif method == 'median':
        return df_returns.median(axis=1)
    else:
        raise ValueError("Metodo non supportato")


def compute_residual_series(target_series: pd.Series, cluster_series: pd.Series) -> pd.Series:
    """
    Regressione del singolo titolo sul cluster → ritorna la serie dei residui
    """
    aligned = pd.concat([target_series, cluster_series], axis=1).dropna()
    y = aligned.iloc[:, 0].values.reshape(-1, 1)
    X = aligned.iloc[:, 1].values.reshape(-1, 1)

    reg = LinearRegression().fit(X, y)
    predicted = reg.predict(X).flatten()
    residuals = aligned.iloc[:, 0] - predicted
    residuals.index = aligned.index
    return residuals


def compute_relative_value_signals(df_returns: pd.DataFrame, z_window=20):
    """
    Calcola segnali di relative value per ogni CCTeu rispetto al cluster medio
    Ritorna un dataframe con z-score dei residui
    """
    cluster_index = compute_cluster_index(df_returns)

    zscores = {}
    for col in df_returns.columns:
        residuals = compute_residual_series(df_returns[col], cluster_index)
        zscores[col] = rolling_zscore(residuals, window=z_window)

    df_zscores = pd.DataFrame(zscores).dropna()
    return df_zscores
