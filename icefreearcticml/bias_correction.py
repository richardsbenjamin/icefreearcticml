from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from icefreearcticml.icefreearcticml.constants import (
    VARIABLES as VAR_NAMES,
    MODELS as _MODELS,
    MODEL_COLOURS,
)


DEFAULT_BIAS_METHODS: Dict[str, dict] = {
    "linear_scaling": {"group": "time.month", "kind": "+"},
    "variance_scaling": {"group": "time.month", "kind": "+"},
    "quantile_mapping": {"n_quantiles": 250, "kind": "+"},
}

DEFAULT_BIAS_VARS: List[str] = ["wsiv", "tas", "oht_atl", "oht_pac"]
# Exclude 'Observations'
DEFAULT_MODEL_NAMES: List[str] = _MODELS[:-1]


def to_da_from_df(df: DataFrame, name: str) -> xr.DataArray:
    return xr.DataArray(
        df.values.T,
        coords={"ensemble": df.columns, "time": df.index},
        dims=["ensemble", "time"],
        name=name,
    )

def get_bias_corrected_members(
    model_df: DataFrame,
    bias_method: str,
    obsh: xr.DataArray,
    simh: xr.DataArray,
    simp: xr.DataArray,
    apply_correction: Callable[..., xr.DataArray],
    method_kwargs: dict,
) -> DataFrame:
    corrected_cols: List[np.ndarray] = []
    for i in model_df.columns:
        out_da = apply_correction(
            method=bias_method,
            obs=obsh,
            simh=simh.sel(ensemble=i),
            simp=simp.sel(ensemble=i),
            **(method_kwargs or {}),
        )
        corrected_cols.append(out_da.values)

    result = DataFrame(corrected_cols).T
    result.index = model_df.index
    result.columns = model_df.columns
    return result


def compute_bias_corrections(
    model_data: dict,
    bias_methods: Dict[str, dict] | None,
    var_names: List[str] | None,
    model_names: List[str] | None,
    apply_correction: Callable[..., xr.DataArray],
) -> Dict[str, Dict[str, Dict[str, DataFrame]]]:
    if bias_methods is None:
        bias_methods = DEFAULT_BIAS_METHODS
    if var_names is None:
        var_names = list(VAR_NAMES)
    if model_names is None:
        model_names = list(DEFAULT_MODEL_NAMES)

    bias_corrections: Dict[str, Dict[str, Dict[str, DataFrame]]] = {}

    for method_name, method_kwargs in bias_methods.items():
        bias_corrections[method_name] = {}
        for var in var_names:
            bias_corrections[method_name][var] = {}
            for model_name in model_names:
                ensemble_df: DataFrame = model_data[var][model_name]
                obsh_s: DataFrame | np.ndarray = model_data[var]["Observations"]
                obsh = xr.DataArray(obsh_s, coords=[obsh_s.index], dims=["time"], name="observations")

                simp = to_da_from_df(ensemble_df, name="all_ensembles")
                # align historical window to observations
                simh = simp.sel(time=obsh.time).rename("historical_ensembles")

                result_df = get_bias_corrected_members(
                    ensemble_df,
                    method_name,
                    obsh,
                    simh,
                    simp,
                    apply_correction,
                    method_kwargs,
                )
                bias_corrections[method_name][var][model_name] = result_df

    return bias_corrections

def score_bias_corrections(
    bias_corrections: Dict[str, Dict[str, Dict[str, DataFrame]]],
    model_data: dict,
    bias_vars: List[str] | None,
    model_names: List[str] | None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], DataFrame]:
    if bias_vars is None:
        bias_vars = DEFAULT_BIAS_VARS
    if model_names is None:
        model_names = list(DEFAULT_MODEL_NAMES)

    bias_correction_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    for var in bias_vars:
        bias_correction_scores[var] = {}
        observations = model_data[var]["Observations"]

        for method_name in bias_corrections.keys():
            bias_correction_scores[var][method_name] = {}

            total = 0.0
            for model in model_names:
                df = bias_corrections[method_name][var][model].loc[observations.index]
                observations_df = DataFrame(
                    np.tile(observations.values.reshape(-1, 1), df.shape[1]),
                    index=df.index,
                    columns=df.columns,
                )
                model_mse = mean_squared_error(df, observations_df)
                bias_correction_scores[var][method_name][model] = model_mse
                total += model_mse

            bias_correction_scores[var][method_name]["all"] = total / len(model_names)

    # Build summary DataFrame (rows: vars, cols: methods)
    res = []
    methods = list(next(iter(bias_corrections.keys())).__iter__()) if bias_corrections else []
    methods = list(bias_corrections.keys())
    for var in bias_vars:
        row = [bias_correction_scores[var][m]["all"] for m in methods]
        res.append(row)

    mse_df = DataFrame(res, index=bias_vars, columns=methods)
    return bias_correction_scores, mse_df


def plot_bias_correction_example(
    model_data: dict,
    bias_corrections: Dict[str, Dict[str, Dict[str, DataFrame]]],
    var: str,
    method: str = "linear_scaling",
    model_names: List[str] | None = None,
    nrows: int = 5,
    figsize: Tuple[int, int] = (24, 32),
):
    if model_names is None:
        model_names = list(DEFAULT_MODEL_NAMES)

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(model_names):
            ax.axis("off")
            continue
        model_name = model_names[i]
        model_color = MODEL_COLOURS[model_name]

        ensemble_df = model_data[var][model_name]
        result_df = bias_corrections[method][var][model_name]
        obsh_s = model_data[var]["Observations"]

        ax.plot(ensemble_df, color=model_color)
        ax.plot(result_df, color="black", linestyle=":")
        ax.plot(obsh_s.index, obsh_s, "k--", linewidth=2, label=model_name)
        ax.set_title(f"{model_name} - {var} ({method})")

    return fig

__all__ = [
    "DEFAULT_BIAS_METHODS",
    "DEFAULT_BIAS_VARS",
    "DEFAULT_MODEL_NAMES",
    "to_da_from_df",
    "get_bias_corrected_members",
    "compute_bias_corrections",
    "score_bias_corrections",
    "plot_bias_correction_example",
]