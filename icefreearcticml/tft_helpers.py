from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pandas import DataFrame, DatetimeIndex
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from icefreearcticml.icefreearcticml.utils import filter_by_years, calculate_first_icefree_year
from icefreearcticml.icefreearcticml.pipeline_helpers import (
    get_datetime_index,
    get_train_test_ensembles,
)


def get_melt(var_data: DataFrame, var: str) -> DataFrame:
    return var_data.reset_index(names="time").melt(id_vars=["time"], var_name="member", value_name=var)


class TrainConfig:

    def __init__(
        self,
        y_var: str,
        x_vars: list[str],
        train_split: float,
        model_name: str,
        **kwargs: dict,
    ) -> None:
        self.y_var = y_var
        self.x_vars = x_vars
        self.train_split = train_split
        self.model_name = model_name
        self.members_for_model = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_emul_test_df(self, model_data: dict) -> DataFrame:
        return filter_by_years(
            model_data[self.y_var][self.model_name],
            self.emul_start, self.emul_end
        )[self.test_members]

    def set_all_data(self, model_data: dict, data: dict | None = None) -> None:
        merged = None
        if data is None:
            data = model_data
        for var in self.all_vars:
            to_filter = data[var][self.model_name].fillna(0)
            if self.members_for_model is not None:
                to_filter = to_filter[self.members_for_model]
            res = get_melt(to_filter, var)
            if merged is None:
                merged = res
            else:
                merged = merged.merge(res, on=["member", "time"])

        merged["time"] = merged["time"].dt.year
        self.all_data = merged
        self.set_simul_datetime_index()
        self.set_train_test_members()

    def set_members_for_model(self, members_for_model: list) -> None:
        self.members_for_model = members_for_model

    def set_simul_datetime_index(self) -> DatetimeIndex:
        self.simul_start = self.all_data["time"].values[0]
        self.simul_end = self.all_data["time"].values[-1]
        start = self.simul_start
        if hasattr(self, "max_encoder_length"):
            start += self.max_encoder_length
        self.simul_index = get_datetime_index(start, self.simul_end)

    def set_emul_data(self) -> None:
        self.emul_df = self.all_data[self.all_data["time"] > (self.simul_end - self.max_encoder_length)]
        self.emul_train = self.emul_df[self.emul_df["member"].isin(self.train_members)]
        self.emul_test = self.emul_df[self.emul_df["member"].isin(self.test_members)]

    def set_simul_data(self) -> None:
        self.simul_df = self.all_data[self.all_data["time"] <= self.simul_end]
        self.simul_train = self.simul_df[self.simul_df["member"].isin(self.train_members)]
        self.simul_test = self.simul_df[self.simul_df["member"].isin(self.test_members)]

    def get_simul_test_df(self) -> DataFrame:
        y_test_simul = self.all_data[self.all_data["time"] >= (self.simul_start + self.max_encoder_length)]
        y_test_simul = y_test_simul.set_index(["time", "member"])["ssie"].unstack()[self.test_members]
        y_test_simul.index = self.simul_index
        return y_test_simul

    def get_test_df(self) -> DataFrame:
        y_test_simul = self.all_data[self.all_data["time"] >= (self.simul_start + self.max_encoder_length)]
        y_test_simul = y_test_simul.set_index(["time", "member"])["ssie"].unstack()[self.test_members]
        y_test_simul.index = self.simul_index
        return y_test_simul

    def set_train_test_members(self) -> None:
        self.n_time_steps = len(self.all_data["time"].unique())
        self.n_members = len(self.all_data["member"].unique())
        train_members, test_members = get_train_test_ensembles(self.n_members, self.train_split)
        self.test_members = self.all_data["member"].unique()[test_members]
        self.train_members = self.all_data["member"].unique()[train_members]

    @property
    def all_vars(self) -> list[str]:
        return [self.y_var, *self.x_vars]

    @property
    def train_data(self) -> DataFrame:
        return self.all_data[self.all_data["member"].isin(self.train_members)]

    @property
    def test_data(self) -> DataFrame:
        return self.all_data[self.all_data["member"].isin(self.test_members)]


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
    from pytorch_forecasting.models import TemporalFusionTransformer
    from lightning.pytorch import Trainer
    model = TemporalFusionTransformer.from_dataset(train_dataloader.dataset)
    trainer = Trainer(max_epochs=15, accelerator="auto", enable_progress_bar=False)
    trainer.fit(model, train_dataloader, val_dataloader)
    return model


def reshape_prediction(prediction: Any, index: list, target: str) -> DataFrame:
    output_df = prediction.index
    output_df[target] = prediction.output.cpu().detach().numpy().ravel()
    output_df = output_df.set_index(["time", "member"])[target].unstack()
    output_df.index = index
    return output_df


class Output:

    def __init__(
        self,
        y_test_simul: DataFrame,
        y_pred_simul: DataFrame,
        train_config: TrainConfig,
        model_res: dict,
    ) -> None:
        self.y_test_simul = y_test_simul
        self.y_pred_simul = y_pred_simul
        self.train_config = train_config
        self.model_res = model_res
        self._simul_resids = None
        self._emul_resids = None

    @property
    def simul_resids(self) -> None:
        if self._simul_resids is None:
            self._simul_resids = (
                (self.y_test_simul - self.y_pred_simul)
                .reset_index(names="time")
                .melt(id_vars=["time"], var_name="member", value_name="resid")
            )
        return self._simul_resids

    @property
    def emul_resids(self) -> None:
        if self._emul_resids is None:
            self._emul_resids = (
                (self.y_test_emul - self.y_pred_emul)
                .reset_index(names="time")
                .melt(id_vars=["time"], var_name="member", value_name="resid")
            )
        return self._emul_resids


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


def plot_liangs_dict(
    original_liang_res: dict,
    exp_outputs_liang_dict: dict,
    liang_config: Any,
    label_str_func: Callable | None = None,
    figsize: tuple = (24, 12),
):
    import matplotlib.pyplot as plt
    from icefreearcticml.pipeline_helpers import plot_liang_tau_avgs, MODEL_COLOURS
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot_liang_tau_avgs(axes[0], original_liang_res, liang_config.x_liang)
    for i, (key, liang_res_emul) in enumerate(exp_outputs_liang_dict.items()):
        plot_liang_tau_avgs(
            axes[1],
            liang_res_emul,
            liang_config.x_liang,
            ["all"],
            colour=list(MODEL_COLOURS.values())[i],
            label_str=label_str_func(key) if label_str_func else None,
        )
    return fig


def get_members_by_percentile(ds: xr.DataSet, var_name: str, p: float, type_: str = "lt"):
    threshold = ds[var_name].quantile(p)
    if type_ == "lt":
        ds_filtered = ds.where(ds[var_name] < threshold, drop=True)
    elif type_ == "gte":
        ds_filtered = ds.where(ds[var_name] >= threshold, drop=True)
    return ds_filtered["ensemble"].values