
import numpy as np
from pandas import DataFrame, Series, concat
from typing import Dict

from icefreearcticml.utils import calculate_ensemble_mean, get_year_list
from icefreearcticml.constants import (
    MODELS,
    MODEL_NAMES,
    MODEL_START_YEAR,
    MODEL_END_YEAR,
    VARIABLES,
    VAR_OBS_START_YEARS,
)

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

def read_model_data(model: str) -> tuple[np.ndarray]:
    ssie, wsie, wsiv, tas, oht_atl, oht_pac, swfd, lwfd = np.load(f'./data/Timeseries_{model}.npy', allow_pickle=True)
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
    
    add_all(model_data)
    return model_data