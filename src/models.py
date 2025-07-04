import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from typing import List, Tuple

def train_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_type: str = "linear"
) -> Tuple:
    """
    Allena un modello di regressione sul target dato.

    model_type: "linear", "ridge", "lasso"
    """

    # Definizione variabili
    X = df[feature_columns]
    y = df[target_column]

    # Selezione modello
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    elif model_type == "lasso":
        model = LassoCV(alphas=[0.001, 0.01, 0.1])
    else:
        raise ValueError("Tipo di modello non supportato")

    model.fit(X, y)

    # Output: modello, predizioni, metriche
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    return model, y_pred, {"R2": r2, "RMSE": rmse}
