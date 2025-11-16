from __future__ import annotations

from typing import List

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from icefreearcticml.icefreearcticml.biascorrectionml.utils import prepare_data


def train_multivariate_randomforest(X_train, y_train, **params):
    model = MultiOutputRegressor(RandomForestRegressor(**params))
    model.fit(X_train, y_train)
    return model

def train_multivariate_linear(X_train, y_train, **params):
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model

def train_multivariate_neuralnet(X_train, y_train, **params):
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_chained_randomforest(X_train, y_train, **params):
    model = RegressorChain(RandomForestRegressor(**params))
    model.fit(X_train, y_train)
    return model

def run_multivariate_bias_correction(
    model_data: dict,
    variables: List[str],
    model_name: str,
    method: str = "randomforest",
    train_split: float = 0.6,
    val_split: float = 0.2,
    **params,
):
    data_res = prepare_data(
        model_data=model_data,
        variables=variables,
        model_name=model_name,
        train_split=train_split,
        val_split=val_split,
    )

    x_train = data_res["x_train"]
    y_train = data_res["y_train"]

    model = METHOD_MAP[method](x_train, y_train, **params)

    return {
        "model": model,
        "data_res": data_res,
    }

METHOD_MAP = {
    "random_forest": train_multivariate_randomforest,
    "linear": train_multivariate_linear,
    "neural_network": train_multivariate_neuralnet,
    "chained_rf": train_chained_randomforest,
}