from __future__ import annotations

import numpy as np
from datetime import datetime
from numpy import datetime64, load, mean, nan
from pandas import DataFrame, Series
from scipy.stats import combine_pvalues, pearsonr, spearmanr, kendalltau

from icefreearcticml._typing import TYPE_CHECKING
from icefreearcticml.constants import (
    MODEL_COLOURS,
    MODELS,
    MODEL_START_YEAR,
    MODEL_END_YEAR,
    MULTI_LIANG_RES_NAMES,
    SIG_LVL,
    VAR_LEGEND_ARGS,
    VAR_YLIMITS,
    VARIABLES,
    VAR_OBS_START_YEARS,
)
from icefreearcticml.liangindex import compute_liang_nvar

if TYPE_CHECKING:
    from icefreearcticml._typing import Axes, ndarray


def calculate_bias(
        obs_data: Series | DataFrame,
        model_data: Series | DataFrame,
        start_year: str,
        end_year: str,
    ) -> float:
    obs_mean = filter_by_years(obs_data, start_year, end_year).mean()
    model_mean = filter_by_years(model_data, start_year, end_year).mean()
    return model_mean - obs_mean

def calculate_combined_pvalues(p_values, var_liangs):
    p_array = np.array(p_values)
    return {
        var: combine_pvalues(p_array[:, i], method='fisher')[1]
            for i, var in enumerate(var_liangs)
    }

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

def calculate_ensemble_max(model_data: ndarray) -> ndarray:
    return model_data.max(axis=1)

def calculate_ensemble_mean(model_data: ndarray) -> ndarray:
    return model_data.mean(axis=1)

def calculate_ensemble_min(model_data: ndarray) -> ndarray:
    return model_data.min(axis=1)

def calculate_first_icefree_year(
        model_ssie: Series | DataFrame,
        threshold: float = 1.0,
    ) -> datetime:
    return (model_ssie < threshold).idxmax()

def calculate_liang_model_flows(liang_data, y, dt, n_iter):
    liang_indexes = []
    p_values = []

    for i in range(y.shape[1]):
        x = prepare_liang_array(liang_data, i, y[:, i])

        res = compute_liang_nvar_wrapper(x, dt, n_iter)

        tau = abs(res["tau"])
        p_value = calculate_pvalue(tau, res["error_tau"])

        liang_indexes.append(tau[:,4])
        p_values.append(p_value[:,4])

    return liang_indexes, p_values

def calculate_pvalue(tau, error_tau):
    z = tau / error_tau
    return np.exp(-0.717 * z - 0.416 * z**2)

def calculate_tau_avg(liang_indexes, var_liangs):
    tau_avg = np.nanmean(np.array(liang_indexes), axis=0)
    return dict(zip(var_liangs, tau_avg))

def compute_liang_nvar_wrapper(x, dt, n_iter):
    res = dict(zip(
        MULTI_LIANG_RES_NAMES,
        compute_liang_nvar(x, dt, n_iter),
    ))
    res["error_T"] = np.nanstd(res["error_T"],axis=0)
    res["error_tau"] = np.nanstd(res["error_tau"],axis=0)
    res["error_R"] = np.nanstd(res["error_R"],axis=0)
    return res

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
        if model_name == "Observations":
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
    ax.axis(xmin=datetime64('1968-01-01'), xmax=datetime64('2102-01-01'), **VAR_YLIMITS[var])
    return ax

def prepare_for_liang(data: DataFrame, start: str, end: str) -> DataFrame:
    filtered = filter_by_years(data, start, end).fillna(0)
    return subtract_ensemble_mean(filtered).to_numpy()

def prepare_liang_array(liang_data, i, y):
    data_list = []
    for var_data in liang_data.values():
        data_list.append(var_data[:, i])  # Access the ith column of each variable's data
    return np.array([*data_list, y])

def read_model_data(model: str) -> tuple[ndarray]:
    ssie, wsie, wsiv, tas, oht_atl, oht_pac, swfd, lwfd = load(f'./data/Timeseries_{model}.npy', allow_pickle=True)
    return ssie, wsie, wsiv, tas, oht_atl, oht_pac, swfd, lwfd

def read_model_data_all() -> dict:
    """_summary_

    Returns
    -------
    dict
        _description_

    Notes
    -----
    It's easier to read the data in with the model names as outer keys
    and the variable names as inner keys, so we do that first, then loop
    over that dictionary to produce the output dictionary; while doing
    this loop we also construct the observational data as Series objects
    indexed by the years, and the model ensemble data as DataFrame objects
    indexed by the years.

    """
    model_data_in = {
        model: dict(zip(VARIABLES, read_model_data(model))) for model in MODELS
    }
    model_data = {}
    for var in VARIABLES:
        model_dict = {}
        for model in MODELS:
            data = model_data_in[model][var]
            if model == "Observations":
                if var in ("oht_atl", "oht_pac"):
                    # The OHT observations are actually reanalyses,
                    # so need to take the ensemble mean
                    data = calculate_ensemble_mean(DataFrame(data.T))
                data = Series(data)
                data.index = get_year_list(VAR_OBS_START_YEARS[var], VAR_OBS_START_YEARS[var]+data.shape[0]-1)
            else:
                data = DataFrame(data.T)
                if model == "CanESM5" and var not in ("oht_atl", "oht_pac"):
                    # Drop last ensemble member so that all CanESM5 variables
                    # have the same number of ensemble members
                    data = data.drop(columns=[49]) 
                data.index = get_year_list(MODEL_START_YEAR, MODEL_END_YEAR)
            model_dict[model] = data
        model_data[var] = model_dict
    
    return model_data

def subtract_ensemble_mean(model_data: DataFrame) -> DataFrame:
    return model_data.subtract(model_data.mean(axis=1), axis=0)

