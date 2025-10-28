from __future__ import annotations

import argparse
from typing import List, Dict

import numpy as np
import xarray as xr
from pandas import DataFrame, concat

from icefreearcticml.constants import (
    MODELS,
    VARIABLES as VAR_NAMES,
)
from icefreearcticml.utils import (
    calculate_bias,
    calculate_first_icefree_year,
    read_model_data_all,
)
from icefreearcticml.pipeline_helpers import add_all


MODEL_NAMES: List[str] = MODELS[:-1]  # exclude 'Observations'
DEFAULT_BIAS_START = "1980-01-01"
DEFAULT_BIAS_END = "2014-01-01"


def compute_ice_free_year_ds(model_data: Dict) -> xr.Dataset:
    model_ensemble_icefree_years = {
        model: calculate_first_icefree_year(model_data["ssie"][model])
        for model in MODEL_NAMES
    }

    df = DataFrame(model_ensemble_icefree_years)
    melted = df.melt().dropna()

    concated = None
    for model in melted["variable"].unique():
        model_years = melted[melted["variable"] == model].reset_index(drop=True)
        model_years["variable"] = model_years["variable"] + "_" + model_years.index.map(str)
        if concated is None:
            concated = model_years
        else:
            concated = concat([concated, model_years])

    ice_free_year_ds = xr.Dataset(
        data_vars={
            "ice_free_years": (("ensemble",), concated["value"].values),
        },
        coords={
            "ensemble": concated["variable"].values,
        },
    )
    return ice_free_year_ds


def compute_bias_ds(
    model_data: Dict,
    variables: List[str],
    bias_start: str = DEFAULT_BIAS_START,
    bias_end: str = DEFAULT_BIAS_END,
) -> xr.Dataset:
    model_ensemble_biases = {
        var: {
            model: calculate_bias(model_data[var]["Observations"], model_data[var][model], bias_start, bias_end)
            for model in MODEL_NAMES
        } for var in variables
    }

    # build coordinates
    first_var = variables[0]
    ensembles = [f"{m}_{i}" for m, data in model_ensemble_biases[first_var].items() for i in range(len(data))]

    # derive years for each ensemble member from first-ice-free-year result
    model_ensemble_icefree_years = {
        model: calculate_first_icefree_year(model_data["ssie"][model])
        for model in MODEL_NAMES
    }
    years = [y for data in model_ensemble_icefree_years.values() for y in data]

    data_vars = {}
    for var in variables:
        values = [v for data in model_ensemble_biases[var].values() for v in data]
        data_vars[f"{var}_bias"] = (("ensemble",), values)
        data_vars[f"{var}_bias_abs"] = (("ensemble",), np.abs(values))

    bias_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "ensemble": ensembles,
            "years": ("ensemble", years),
        },
    )
    return bias_ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute bias and ice-free-year datasets")
    p.add_argument("--bias-start", default=DEFAULT_BIAS_START, help="Bias baseline start date (YYYY-MM-DD)")
    p.add_argument("--bias-end", default=DEFAULT_BIAS_END, help="Bias baseline end date (YYYY-MM-DD)")
    p.add_argument("--vars", default=",".join(VAR_NAMES), help="Comma-separated variable names to include")
    p.add_argument("--save-dir", default=None, help="Optional directory to save outputs as NetCDF")
    return p.parse_args()


def maybe_save(ds: xr.Dataset, path: str | None) -> None:
    if path is None:
        return
    ds.to_netcdf(path)


def main() -> None:
    args = parse_args()
    variables = [v for v in args.vars.split(",") if v]

    # load and prepare data
    model_data = read_model_data_all()
    add_all(model_data)

    # compute outputs
    ice_free_year_ds = compute_ice_free_year_ds(model_data)
    bias_ds = compute_bias_ds(model_data, variables, args.bias_start, args.bias_end)

    # optional save
    if args.save_dir:
        ice_path = f"{args.save_dir.rstrip('/')}/ice_free_year_ds.nc"
        bias_path = f"{args.save_dir.rstrip('/')}/bias_ds.nc"
        maybe_save(ice_free_year_ds, ice_path)
        maybe_save(bias_ds, bias_path)
        print(f"Saved: {ice_path}\nSaved: {bias_path}")
    else:
        # print brief summaries
        print("ice_free_year_ds:", ice_free_year_ds)
        print("bias_ds:", bias_ds)


if __name__ == "__main__":
    main()