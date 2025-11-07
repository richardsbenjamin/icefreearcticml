from __future__ import annotations

import numpy as np
from pandas import DataFrame
from scipy.stats import combine_pvalues

from icefreearcticml.constants import (
    MULTI_LIANG_RES_NAMES,
)
from icefreearcticml.liangindex import compute_liang_nvar
from icefreearcticml.utils import filter_by_years, subtract_ensemble_mean



def calculate_all_liang_flows(model_data: dict, liang_config: LiangConfig, model_names: list = MODEL_NAMES) -> dict:
    liang_start = liang_config.liang_start
    liang_end = liang_config.liang_end
    var_liangs = liang_config.x_liang
    y_var = liang_config.y_liang
    dt = liang_config.dt
    n_iter = liang_config.n_iter
    tau_avgs = {}
    combined_pvalues = {}
    ensemble_flows = {}
    ensemble_pvalues = {}
    for model in model_names:
        ssie_liang = prepare_for_liang(model_data[y_var][model], liang_start, liang_end)
        liang_data = {var: prepare_for_liang(model_data[var][model], liang_start, liang_end) for var in var_liangs}
        liang_indexes, p_values = calculate_liang_model_flows(liang_data, ssie_liang, dt, n_iter)
        ensemble_flows[model] = np.array(liang_indexes)
        ensemble_pvalues[model] = p_values
        tau_avgs[model] = calculate_tau_avg(liang_indexes, var_liangs)
        combined_pvalues[model] = calculate_combined_pvalues(p_values, var_liangs)
    return {
        "tau_avgs": tau_avgs,
        "combined_pvalues": combined_pvalues,
        "ensemble_flows": ensemble_flows,
        "ensemble_pvalues": ensemble_pvalues,
    }

def calculate_combined_pvalues(p_values, var_liangs):
    p_array = np.array(p_values)
    return {
        var: combine_pvalues(p_array[:, i], method='fisher')[1]
            for i, var in enumerate(var_liangs)
    }

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

def get_liangs_from_output_dict(output_dict: dict[Output], liang_config: LiangConfig, model_names: list) -> dict:
    liangs = {}
    for key, outputs in output_dict.items():
        emulated_model_data = ModelData({"all": outputs})
        liang_res_emulated = calculate_all_liang_flows(emulated_model_data, liang_config, model_names=model_names)
        liangs[key] = liang_res_emulated
    return liangs

def plot_liangs_dict(
    original_liang_res: dict,
    exp_outputs_liang_dict: dict,
    liang_config: Any,
    label_str_func: Callable | None = None,
    figsize: tuple = (24, 12),
):
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

def plot_liang_tau_avgs(ax: Any, liang_res, var_liangs, model_names: list = MODEL_NAMES):
    ax.grid(linestyle='--')
    jitter = 0.1
    for j, var in enumerate(var_liangs):
        for i, model in enumerate(model_names):
            tau = liang_res["tau_avgs"][model][var]
            pvalue = liang_res["combined_pvalues"][model][var]
            x = j - 0.2 + i * jitter
            kwargs = {"edgecolors": 'black', "linewidths": 2} if pvalue <= 0.05 else {}
            ax.scatter(x, tau, label=model, c=MODEL_COLOURS[model], s=100, **kwargs)
    ax.set_xticks(range(len(var_liangs)))
    ax.set_xticklabels([VAR_YLABELS_SHORT[var] for var in var_liangs])
    ax.set_ylabel("Information Transfer")
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylim(-1, 30)
    ax.grid(linestyle='--')
    ax.set_axisbelow(True)

def prepare_for_liang(data: DataFrame, start: str, end: str) -> DataFrame:
    filtered = filter_by_years(data, start, end).fillna(0)
    return subtract_ensemble_mean(filtered).to_numpy()

def prepare_liang_array(liang_data, i, y):
    data_list = []
    for var_data in liang_data.values():
        data_list.append(var_data[:, i])  # Access the ith column of each variable's data
    return np.array([*data_list, y])


class LiangConfig:
    def __init__(
        self,
        liang_start: str,
        liang_end: str,
        x_liang: list,
        y_liang: str,
        dt: int = 1,
        n_iter: int = 1000,
    ) -> None:
        self.liang_start = liang_start
        self.liang_end = liang_end
        self.x_liang = x_liang
        self.y_liang = y_liang
        self.dt = dt
        self.n_iter = n_iter

