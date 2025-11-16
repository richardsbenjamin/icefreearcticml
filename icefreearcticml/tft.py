from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from icefreearcticml.icefreearcticml.utils.output import Output
from icefreearcticml.icefreearcticml.utils.trainconfig import TrainConfig
from icefreearcticml.icefreearcticml.utils.utils import (
    calculate_first_icefree_year,
)
from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer


def get_timeseries_dataloader(data: DataFrame, train_config: TrainConfig, train: bool):
    from pytorch_forecasting import TimeSeriesDataSet
    dataset = TimeSeriesDataSet(
        data,
        time_idx="time",
        group_ids=["member"],
        target=train_config.y_var,
        max_encoder_length=train_config.max_encoder_length,
        max_prediction_length=train_config.max_prediction_length,
        time_varying_known_reals=train_config.x_vars,
    )
    return dataset.to_dataloader(train=train, batch_size=32, num_workers=0)

def get_timeseries_model(train_dataloader, val_dataloader):
    model = TemporalFusionTransformer.from_dataset(train_dataloader.dataset)
    trainer = Trainer(max_epochs=20, accelerator="auto", enable_progress_bar=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    return model

def reshape_prediction(prediction: Any, index: list, target: str) -> DataFrame:
    output_df = prediction.index
    output_df[target] = prediction.output.cpu().detach().numpy().ravel()
    output_df = output_df.set_index(["time", "member"])[target].unstack()
    output_df.index = index
    return output_df

def run_model_from_config(train_config: TrainConfig, model_data: dict, data: dict | None = None) -> Output:
    train_config.set_all_data(model_data, data)
    train_dataloader = get_timeseries_dataloader(train_config.train_data, train_config, train=True)
    val_dataloader = get_timeseries_dataloader(train_config.test_data, train_config, train=False)
    model = get_timeseries_model(train_dataloader, val_dataloader)
    y_test_df = train_config.get_simul_test_df()
    simulations = model.predict(val_dataloader, return_index=True)
    simulations_df = reshape_prediction(simulations, y_test_df.index, target=train_config.y_var)
    return Output(
        y_test_df,
        simulations_df,
        train_config,
        model.state_dict(),
    )

def calculate_trend_correlation(original_data, emulated_data, time_axis):
    original_trends = []
    emulated_trends = []
    for i in range(original_data.shape[1]):
        orig_slope = np.polyfit(time_axis, original_data[:, i], 1)[0]
        original_trends.append(orig_slope)
        emul_slope = np.polyfit(time_axis, emulated_data[:, i], 1)[0]
        emulated_trends.append(emul_slope)
    original_trends = np.array(original_trends)
    emulated_trends = np.array(emulated_trends)
    trend_correlation, p_value = pearsonr(original_trends, emulated_trends)
    return trend_correlation, p_value, original_trends, emulated_trends

def get_bias_impact_row(output: Output) -> dict:
    ice_free_years = calculate_first_icefree_year(output.y_pred_simul).dt.year.values
    ify_mean = ice_free_years.mean()
    ify_std = ice_free_years.std()
    rmse = np.sqrt(mean_squared_error(output.y_test_simul, output.y_pred_simul))
    r2 = r2_score(output.y_test_simul, output.y_pred_simul)
    trend_corr = calculate_trend_correlation(output.y_test_simul.values, output.y_pred_simul.values, np.arange(len(output.y_test_simul)))
    return {
        "Ice Free Year Mean": ify_mean,
        "Ice Free Year Std": ify_std,
        "RMSE": rmse,
        "R2": r2,
        "Trend Correlation": trend_corr[0],
    }

def get_members_by_percentile(ds: xr.DataSet, var_name: str, p: float, type_: str = "lt"):
    threshold = ds[var_name].quantile(p)
    if type_ == "lt":
        ds_filtered = ds.where(ds[var_name] < threshold, drop=True)
    elif type_ == "gte":
        ds_filtered = ds.where(ds[var_name] >= threshold, drop=True)
    return ds_filtered["ensemble"].values

