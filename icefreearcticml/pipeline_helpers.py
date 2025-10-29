from __future__ import annotations

import random
from typing import Tuple, List, Dict, Any

import numpy as np
from pandas import DataFrame, DatetimeIndex, Timestamp, concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import combine_pvalues

from icefreearcticml.icefreearcticml.constants import (
    MODELS,
    MODEL_COLOURS,
    MULTI_LIANG_RES_NAMES,
    VAR_YLABELS_SHORT,
)
from icefreearcticml.icefreearcticml.utils import (
    calculate_first_icefree_year,
    filter_by_years,
)
from icefreearcticml.icefreearcticml.liangindex.function_liang_nvar import compute_liang_nvar

# Module-level handle to notebook-loaded data to minimize notebook refactors
MODEL_DATA: Dict | None = None
MODEL_NAMES = MODELS[:-1]


def set_model_data(model_data: Dict) -> None:
    global MODEL_DATA
    MODEL_DATA = model_data


# ----------------------
# Data helpers
# ----------------------

def add_all(model_data: Dict) -> None:
    for var in model_data.keys():
        frames = [model_data[var][m] for m in MODEL_NAMES]
        all_ = concat(frames, axis=1)
        col_names = []
        for m in MODEL_NAMES:
            ncols = model_data[var][m].shape[-1]
            col_names += [f"{m}_{i}" for i in range(ncols)]
        all_.columns = col_names
        model_data[var]["all"] = all_

def get_datetime_index(start_year: int, end_year: int) -> DatetimeIndex:
    dt_list = [f"{year}-01-01" for year in range(start_year, end_year + 1)]
    return DatetimeIndex(dt_list)


def get_x(train_config: Any, period: str, detrend: bool = False):
    assert MODEL_DATA is not None, "MODEL_DATA not set. Call set_model_data(model_data)."
    if period in ("simul", "simulation"):
        start, end = train_config.simul_start, train_config.simul_end
    elif period in ("emul", "emulation"):
        start, end = train_config.emul_start, train_config.emul_end
    xs = []
    ensemble_means = {}
    for x_var in train_config.x_vars:
        res = filter_by_years(MODEL_DATA[x_var][train_config.model_name], start, end)
        if detrend:
            ensemble_mean = res.mean(axis=1)
            ensemble_means[x_var] = ensemble_mean
            res = res.subtract(ensemble_mean, axis=0)
        xs.append(res)
    if detrend:
        return np.stack(xs), ensemble_means
    return np.stack(xs), None


def get_x_train_test(x: np.ndarray, train_config: Any) -> Tuple[np.ndarray, np.ndarray]:
    X_train = x[:, :, train_config.train_members].reshape(-1, len(train_config.x_vars))
    X_test = x[:, :, train_config.test_members].reshape(-1, len(train_config.x_vars))
    return X_train, X_test


def get_reshaped(x: np.ndarray, nt: int, ensembles: List[int]) -> DataFrame:
    return DataFrame(x.reshape((nt, len(ensembles))), columns=ensembles)


def get_train_test_ensembles(n: int, train_split: float) -> Tuple[List[int], List[int]]:
    n_train = int(n * train_split)
    train_ensembles = select_ensembles(n, n_train)
    test_ensembles = select_remaining(n, train_ensembles)
    return train_ensembles, test_ensembles


def select_ensembles(n: int, x: int) -> List[int]:
    return random.sample(range(n), x)


def select_remaining(n: int, selected_ensembles: List[int]) -> List[int]:
    return [i for i in range(n) if i not in selected_ensembles]


def get_y(train_config: Any, period: str, detrend: bool = False):
    assert MODEL_DATA is not None, "MODEL_DATA not set. Call set_model_data(model_data)."
    if period in ("simul", "simulation"):
        start, end = train_config.simul_start, train_config.simul_end
    elif period in ("emul", "emulation"):
        start, end = train_config.emul_start, train_config.emul_end
    res = filter_by_years(MODEL_DATA[train_config.y_var][train_config.model_name], start, end)
    if detrend:
        ensemble_mean = res.mean(axis=1)
        res = res.subtract(ensemble_mean, axis=0)
        return res.to_numpy(), ensemble_mean
    return res.to_numpy(), None


def get_y_train_test(y: np.ndarray, train_config: Any) -> Tuple[np.ndarray, np.ndarray]:
    y_train = y[:, train_config.train_members].flatten()
    y_test = y[:, train_config.test_members].flatten()
    return y_train, y_test


class Output:
    def __init__(
        self,
        y_test_emul: DataFrame,
        y_pred_emul: DataFrame,
        y_test_simul: DataFrame,
        y_pred_simul: DataFrame,
        train_config: DataFrame,
        model_res: dict,
    ) -> None:
        self.y_test_emul = y_test_emul
        self.y_pred_emul = y_pred_emul
        self.y_test_simul = y_test_simul
        self.y_pred_simul = y_pred_simul
        self.train_config = train_config
        self.model_res = model_res


# ----------------------
# Model helpers
# ----------------------

def get_emulated(model: Any, train_config: Any) -> DataFrame:
    X_emul, _ = get_x(train_config, "emul")
    _, X_emul_test = get_x_train_test(X_emul, train_config)
    y_emul = model.predict(X_emul_test)
    y_emul_df = get_reshaped(y_emul, X_emul.shape[1], train_config.test_members)
    y_emul_df.index = get_datetime_index(train_config.emul_start, train_config.emul_end)
    return y_emul_df


def get_emulated_ice_free_years(train_config: Any, n_iter: int) -> List[int]:
    X, _ = get_x(train_config, "simul")
    ice_free_years_emul: List[int] = []
    for _ in range(n_iter):
        train_config.set_train_test_members(X.shape)
        y, _ = get_y(train_config, "simul")
        X_train, X_test = get_x_train_test(X, train_config)
        y_train, y_test = get_y_train_test(y, train_config)
        reg_res = get_regression(X_train, y_train, X_test, y_test)
        y_pred_emul = get_emulated(reg_res["model"], train_config)
        ice_free_years_emul.extend(
            calculate_first_icefree_year(y_pred_emul).dt.year.to_list()
        )
    return ice_free_years_emul


def get_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: List[float] | None = None,
    **rf_kwargs: dict,
) -> dict:
    model = RandomForestRegressor(**rf_kwargs)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(x_test)
    return {
        "y_pred": y_pred,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "model": model,
    }


def get_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: List[float] | None = None,
) -> dict:
    model = LinearRegression()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(x_test)
    return {
        "y_pred": y_pred,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "model": model,
    }


def get_y_emulated_outputs(model: Any, train_config: Any) -> tuple[DataFrame, DataFrame]:
    assert MODEL_DATA is not None, "MODEL_DATA not set. Call set_model_data(model_data)."
    y_pred_emul = get_emulated(model, train_config).clip(0)
    y_test_emul = filter_by_years(
        MODEL_DATA[train_config.y_var][train_config.model_name],
        train_config.emul_start,
        train_config.emul_end,
    )[train_config.test_members]
    return y_test_emul, y_pred_emul


def get_y_simulated_outputs(y_test: np.ndarray, y_pred: np.ndarray, train_config: Any) -> tuple[DataFrame, DataFrame]:
    simul_index = get_datetime_index(train_config.simul_start, train_config.simul_end)
    y_test_simul = get_reshaped(y_test, train_config.n_time_steps, train_config.test_members)
    y_test_simul.index = simul_index
    y_pred_simul = get_reshaped(y_pred, train_config.n_time_steps, train_config.test_members)
    y_pred_simul.index = simul_index
    return y_test_simul, y_pred_simul


# ----------------------
# Plotting
# ----------------------

def plot_emulation(ax: Any, output: Output, model_type: str) -> None:
    model_color = MODEL_COLOURS[output.train_config.model_name]
    ax.plot(output.y_test_simul, color=model_color, linewidth=2)
    ax.plot(output.y_pred_simul, color="black", linestyle="--", linewidth=2)
    ax.plot(output.y_test_emul, color=model_color, linewidth=2)
    ax.plot(output.y_pred_emul, color="black", linestyle="--", linewidth=2)
    ax.plot([], [], color="black", linestyle="--", label="Model Prediction")
    ax.plot([], [], color=model_color, label="Ensemble Data")
    ax.axhline(y=1, color='black', linestyle=':', linewidth=2)
    ax.axvline(x=Timestamp(f"{output.train_config.simul_end}-01-01"), color='black', linestyle=':', linewidth=2)
    ax.set_title(f"{output.train_config.model_name} - {output.train_config.y_var} {model_type}", x=0.5, size=20)
    ax.tick_params(labelsize=20)


def plot_ice_free_year_dist(ax: Any, model_name: str, emulated_years: list, data_years: list) -> None:
    ax.hist(emulated_years, bins=50, density=True, label="Emulated")
    ax.hist(data_years, bins=50, density=True, label="Historic")
    ax.legend()
    ax.set_title(model_name)


# ----------------------
# Liang causality
# ----------------------

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


def calculate_combined_pvalues(p_values, var_liangs):
    p_array = np.array(p_values)
    return {var: combine_pvalues(p_array[:, i], method='fisher')[1] for i, var in enumerate(var_liangs)}


def calculate_liang_model_flows(liang_data, y, dt, n_iter):
    liang_indexes = []
    p_values = []
    for i in range(y.shape[1]):
        x = prepare_liang_array(liang_data, i, y[:, i])
        res = compute_liang_nvar_wrapper(x, dt, n_iter)
        tau = abs(res["tau"])
        p_value = calculate_pvalue(tau, res["error_tau"])
        liang_indexes.append(tau[:, 4])
        p_values.append(p_value[:, 4])
    return liang_indexes, p_values


def calculate_pvalue(tau, error_tau):
    z = tau / error_tau
    return np.exp(-0.717 * z - 0.416 * z ** 2)


def calculate_tau_avg(liang_indexes, var_liangs):
    tau_avg = np.nanmean(np.array(liang_indexes), axis=0)
    return dict(zip(var_liangs, tau_avg))


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


def compute_liang_nvar_wrapper(x, dt, n_iter):
    res = dict(zip(MULTI_LIANG_RES_NAMES, compute_liang_nvar(x, dt, n_iter)))
    res["error_T"] = np.nanstd(res["error_T"], axis=0)
    res["error_tau"] = np.nanstd(res["error_tau"], axis=0)
    res["error_R"] = np.nanstd(res["error_R"], axis=0)
    return res


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


def prepare_liang_array(liang_data, i, y):
    data_list = []
    for var_data in liang_data.values():
        data_list.append(var_data[:, i])
    return np.array([*data_list, y])


def prepare_for_liang(data: DataFrame, start: str, end: str) -> DataFrame:
    filtered = filter_by_years(data, start, end).fillna(0)
    return subtract_ensemble_mean(filtered).to_numpy()


def get_liangs_from_output_dict(output_dict: dict[Output], liang_config: LiangConfig, model_names: list) -> dict:
    liangs = {}

    for key, outputs in output_dict.items():
        emulated_model_data = ModelData({"all": outputs})
        liang_res_emulated = calculate_all_liang_flows(emulated_model_data, liang_config, model_names=model_names)
        liangs[key] = liang_res_emulated

    return liangs


def subtract_ensemble_mean(model_data: DataFrame) -> DataFrame:
    return model_data.subtract(model_data.mean(axis=1), axis=0)
