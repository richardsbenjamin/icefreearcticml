
import numpy as np
from pandas import DataFrame, Series, concat
from typing import Dict

from icefreearcticml.icefreearcticml.utils.utils import calculate_ensemble_mean, get_year_list
from icefreearcticml.icefreearcticml.constants import (
    MODELS,
    MODEL_NAMES,
    MODEL_PATH,
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

def read_model_data(model_path: str, model: str) -> tuple[np.ndarray]:
    return np.load(f'{model_path}/Timeseries_{model}.npy', allow_pickle=True)

def read_model_data_all(model_path: str = MODEL_PATH) -> dict:
    model_data_in = {
        model: dict(zip(VARIABLES, read_model_data(model_path, model))) for model in MODELS
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