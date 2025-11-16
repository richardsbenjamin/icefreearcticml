from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from icefreearcticml.icefreearcticml.utils.utils import extend_and_fill_series


# Registry of basic regressors used by uni/multi modules
REGRESSORS: Dict[str, object] = {
    "random_forest": RandomForestRegressor,
    "linear": LinearRegression,
    "gradient_boosting": GradientBoostingRegressor,
    "neural_network": MLPRegressor,
}

def get_regressor(regressor_name: str, **kwargs) -> object:
    if regressor_name not in REGRESSORS:
        raise ValueError(
            f"Unknown regressor: {regressor_name}. Available: {list(REGRESSORS.keys())}"
        )
    return REGRESSORS[regressor_name](**kwargs)

def prepare_data(
    model_data: dict,
    variables: List[str],
    model_name: str,
    train_split: float = 0.6,
    val_split: float = 0.2,
) -> Dict[str, object]:
    if isinstance(variables, str):
        variables = [variables]

    all_members = list(model_data[variables[0]][model_name].columns)
    n_total = len(all_members)

    train_count = int(n_total * train_split)
    val_count = int(n_total * val_split)

    train_members = all_members[:train_count]
    val_members = all_members[train_count : train_count + val_count]
    test_members = all_members[train_count + val_count :]

    x_train: List[np.ndarray] = []
    x_val: List[np.ndarray] = []
    x_test: List[np.ndarray] = []
    y_train: List[np.ndarray] = []
    y_val: List[np.ndarray] = []
    y_test: List[np.ndarray] = []

    for var in variables:
        obs_series = extend_and_fill_series(model_data[var]["Observations"])  # type: ignore[index]
        member_data: DataFrame = model_data[var][model_name].loc[obs_series.index].fillna(0)  # type: ignore[index]

        # Features (model data)
        x_train.append(member_data[train_members].melt()["value"].values)
        x_val.append(member_data[val_members].melt()["value"].values)
        x_test.append(member_data[test_members].melt()["value"].values)

        # Targets (observations)
        y_train.append(np.tile(obs_series.values, len(train_members)))
        y_val.append(np.tile(obs_series.values, len(val_members)))
        y_test.append(np.tile(obs_series.values, len(test_members)))

    # Stack variables: shape = (n_samples * n_members, n_variables)
    x_train = np.column_stack(x_train) if x_train else np.empty((0, 0))
    x_val = np.column_stack(x_val) if x_val else np.empty((0, 0))
    x_test = np.column_stack(x_test) if x_test else np.empty((0, 0))

    y_train_arr = np.column_stack(y_train) if y_train else np.empty((0, 0)).reshape(-1, 1)
    y_val_arr = np.column_stack(y_val) if y_val else np.empty((0, 0)).reshape(-1, 1)
    y_test_arr = np.column_stack(y_test) if y_test else np.empty((0, 0)).reshape(-1, 1)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train_arr,
        "y_val": y_val_arr,
        "y_test": y_test_arr,
        "train_members": train_members,
        "val_members": val_members,
        "test_members": test_members,
    }

