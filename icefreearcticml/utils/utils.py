from __future__ import annotations

import random

import numpy as np
from datetime import datetime
from numpy import mean, nan
from pandas import DataFrame, DatetimeIndex, Index, Series, Timestamp, to_datetime
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

from icefreearcticml.icefreearcticml._typing import TYPE_CHECKING
from icefreearcticml.icefreearcticml.constants import (
    MODEL_COLOURS,
    MODELS,
    SIG_LVL,
    VAR_LEGEND_ARGS,
    VAR_YLIMITS,
    VARIABLES,
)

if TYPE_CHECKING:
    from icefreearcticml._typing import Axes

def calculate_bias(
        obs_data: Series | DataFrame,
        model_data: Series | DataFrame,
        start_year: str,
        end_year: str,
    ) -> float:
    obs_mean = filter_by_years(obs_data, start_year, end_year).mean()
    model_mean = filter_by_years(model_data, start_year, end_year).mean()
    return model_mean - obs_mean

def calculate_correlation_ensemble_mean(
        x_df: DataFrame,
        y_df: DataFrame,
        corr_type: str = "pearson",
        sig_lvl: float = SIG_LVL,
    ) -> float:
    if corr_type == "pearson":
        correlation_func = pearsonr
    elif corr_type == "spearman":
        correlation_func = spearmanr
    else:
        correlation_func = kendalltau 

    corrs = []
    for col in x_df.columns:
        res = correlation_func(x_df[col], y_df[col])
        if res.pvalue < sig_lvl:
            corrs.append(res.statistic)

    return mean(corrs) if corrs else nan

def calculate_ensemble_max(model_data: np.ndarray) -> np.ndarray:
    return model_data.max(axis=1)

def calculate_ensemble_mean(model_data: np.ndarray) -> np.ndarray:
    return model_data.mean(axis=1)

def calculate_ensemble_min(model_data: np.ndarray) -> np.ndarray:
    return model_data.min(axis=1)

def calculate_first_icefree_year(
        model_ssie: Series | DataFrame,
        threshold: float = 1.0,
    ) -> datetime:
    return (model_ssie < threshold).idxmax()

def extend_and_fill_series(
    s: Series,
    start_year: int = 1979,
    end_year: int = 2023,
    min_points: int = 3,
) -> Series:
    years = (
        to_datetime(s.index).year
        if isinstance(s.index[0], (np.datetime64, Timestamp))
        else s.index.astype(int)
    )
    s = Series(s.values, index=years)

    s = s[(s.index >= start_year) & (s.index <= end_year)]

    full_years = Index(range(start_year, end_year + 1))
    s = s.reindex(full_years)

    known = s.dropna()

    if len(known) < min_points:
        s = s.interpolate(limit_direction="both")
    else:
        X = known.index.values.reshape(-1, 1)
        y = known.values
        model = LinearRegression().fit(X, y)
        pred = model.predict(s.index.values.reshape(-1, 1))
        s[np.isnan(s)] = pred[np.isnan(s)]

    s.index = to_datetime(s.index.astype(str) + "-01-01")
    return s

def filter_by_years(
        model_data: Series | DataFrame,
        start_year: str,
        end_year: str,
    ) -> Series | DataFrame:
    if isinstance(start_year, int):
        start_year = f'{start_year}-01-01'
    if isinstance(end_year, int):
        end_year = f'{end_year}-01-01'
    return model_data.loc[start_year:end_year].copy()

def get_datetime_index(start_year: int, end_year: int) -> DatetimeIndex:
    dt_list = [f"{year}-01-01" for year in range(start_year, end_year + 1)]
    return DatetimeIndex(dt_list)

def get_melt(var_data: DataFrame, var: str) -> DataFrame:
    return var_data.reset_index(names="time").melt(id_vars=["time"], var_name="member", value_name=var)

def get_train_test_ensembles(n: int, train_split: float) -> tuple[list]:
    n_train = int(n * train_split)
    n_test = n - n_train

    train_ensembles = select_ensembles(n, n_train)
    test_ensembles = select_remaining(n, train_ensembles)

    return train_ensembles, test_ensembles

def get_shape_df(model_data: dict) -> DataFrame:
    df_in = []
    for var in VARIABLES:
        df_in.append({
            model: model_data[var][model].shape for model in MODELS
        })
    data_shapes = DataFrame(df_in)
    data_shapes.index = VARIABLES
    return data_shapes

def get_year_list(start_year: int, end_year: int) -> list[datetime]:
    return [datetime(year, 1, 1) for year in range(start_year, end_year+1)]

def plot_variable(ax: Axes, var: str, all_var_data: dict, ylabel: str, title_i: int) -> None:
    for i, (model_name, var_data) in enumerate(all_var_data.items()):
        if model_name == "all":
            continue
        elif model_name == "Observations":
            ax.plot(var_data.index, var_data,'k--', linewidth=4, label=model_name)
        else:
            ax.plot(
                var_data.index, calculate_ensemble_mean(var_data), '-',
                color=MODEL_COLOURS[model_name], linewidth=4, label=model_name,
            )
            ax.fill_between(
                var_data.index, calculate_ensemble_min(var_data),
                calculate_ensemble_max(var_data), color=MODEL_COLOURS[model_name], alpha=0.1,
            )
    ax.grid(linestyle='--')
    ax.set_ylabel(ylabel, fontsize=26)
    ax.set_title(chr(ord('a')+title_i),loc='left',fontsize=30,fontweight='bold')
    ax.tick_params(labelsize=20)
    ax.legend(**VAR_LEGEND_ARGS[var])
    ax.axis(xmin=np.datetime64('1968-01-01'), xmax=np.datetime64('2102-01-01'), **VAR_YLIMITS[var])
    return ax

def select_ensembles(n: int, x: int) -> list[int]:
    return random.sample(range(n), x)

def select_remaining(n: int, selected_ensembles: list[int]) -> list[int]:
    return [i for i in range(n) if i not in selected_ensembles]

def subtract_ensemble_mean(model_data: DataFrame) -> DataFrame:
    return model_data.subtract(model_data.mean(axis=1), axis=0)