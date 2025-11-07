from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score

from .utils import get_regressor, prepare_data


def run_regression(
    model_data: dict,
    var: str,
    model_name: str,
    regressor_name: str,
    regressor_params: Optional[Dict] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Dict[str, object]:
    data_res = prepare_data(
        model_data=model_data,
        variables=[var],
        model_name=model_name,
        train_split=train_split,
        val_split=val_split,
    )
    regressor_params = regressor_params or {}
    model = get_regressor(regressor_name, **regressor_params)

    model.fit(data_res["x_train"], data_res["y_train"])

    y_test = data_res["y_test"]     
    x_test = data_res["x_test"]     
    x_val = data_res["x_val"]      
    y_pred_val = model.predict(x_val)
    y_pred = model.predict(x_test)

    val_members = len(data_res["val_members"]) 
    test_members = len(data_res["test_members"]) 

    obs_series = data_res["obs_series"] 
    n = obs_series.shape[0]
    y_pred_df = DataFrame(y_pred.reshape(n, test_members), index=obs_series.index)
    y_pred_val_df = DataFrame(y_pred_val.reshape(n, val_members), index=obs_series.index)
    x_test_df = DataFrame(x_test.reshape(n, test_members), index=obs_series.index)

    return {
        "y_pred_df": y_pred_df,
        "y_pred_val_df": y_pred_val_df,
        "x_test_df": x_test_df,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "model": model,
        "data_res": data_res,
    }

