from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from pandas import DataFrame
from scipy import stats

from icefreearcticml.icefreearcticml.constants import MODEL_NAMES
from icefreearcticml.icefreearcticml.utils.utils import (
    calculate_ensemble_mean,
    filter_by_years,
)


def all_subperiods_np(start:int, end:int, min_len:int=5):
    years = np.arange(start, end+1)
    starts, ends = np.triu_indices(len(years), k=1)
    periods = list(zip(years[starts], years[ends]))
    return [(int(s), int(e)) for (s, e) in periods if (e - s + 1) >= min_len]

def get_em_model(model_type: str, *args: tuple) -> BaseConstraintModel:
    if model_type in EM_MODEL_TYPES:
        return EM_MODEL_TYPES[model_type](*args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
def get_inter_model_df(model_data: dict, var: str, model_names: list[str] = MODEL_NAMES) -> DataFrame:
    res = {}
    for model_name in model_names:
        res[model_name] = calculate_ensemble_mean(model_data[var][model_name].fillna(0))
    return DataFrame(res)

def mean_over_window(df: DataFrame, start_year: int, end_year: int) -> np.ndarray:
    sub = filter_by_years(df, start_year, end_year)
    return sub.to_numpy().mean(axis=0)  # per-member mean

def pick_optimal_hist_window(
    hist_df: DataFrame,
    fut_df: DataFrame,
    obs_start: int,
    obs_end: int,
    proj_start: int,
    proj_end: int,
) -> Tuple[Tuple[int, int], float, float]:
    # target vector across members for this projection window
    Y = mean_over_window(fut_df, proj_start, proj_end)

    best_r = -np.inf
    best_win = (obs_start, obs_end)

    periods = all_subperiods_np(obs_start, obs_end, min_len=5)

    for (hs, he) in periods:
        X = mean_over_window(hist_df, hs, he)
        # correlation across members
        r = abs(float(np.corrcoef(X, Y)[0, 1]))
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_win = (hs, he)

    return best_win, best_r

def run_time_varying_ec(model_data: Dict, hist_var: str, args: argparse.Namespace) -> DataFrame:
    hist_df = get_inter_model_df(model_data, hist_var)
    fut_df = get_inter_model_df(model_data, args.fut_var)
    obs_series: DataFrame = model_data[hist_var]["Observations"]
    rows = []
    # slide projection windows
    for ps in range(args.calib_start, args.calib_end - args.window + 2):
        pe = ps + args.window - 1

        # choose optimal historical window by inter-model correlation
        (hs, he), r = pick_optimal_hist_window(
            hist_df=hist_df,
            fut_df=fut_df,
            obs_start=args.obs_start,
            obs_end=args.obs_end,
            proj_start=ps,
            proj_end=pe,
        )

        # compute needed statistics
        n = fut_df.shape[-1]
        Y = mean_over_window(fut_df, ps, pe)
        X = mean_over_window(hist_df, hs, he)
        Ybar = float(Y.mean())
        Xbar = float(X.mean())

        model = get_em_model(args.model_type, Xbar, Ybar)
        model.fit(X, Y)
        Xo = float(mean_over_window(obs_series, hs, he))
        Yc = model.get_y_c(Xo)

        sigma, ci_low, ci_high = model.get_ci(Y, n, r)
        rows.append({
            "proj_start": ps,
            "proj_end": pe,
            "hist_start": hs,
            "hist_end": he,
            "r": r,
            "raw_mean": Ybar,
            "Yc": Yc,
            "Xo": Xo,
            "sigma": sigma,
            "model_res": model.get_model_params(),
            "ci_low": ci_low,
            "ci_high": ci_high,
            **{f"X_{model_name}": x for model_name, x in zip(MODEL_NAMES, X)},
            **{f"Y_{model_name}": y for model_name, y in zip(MODEL_NAMES, Y)}
        })
    return DataFrame(rows)


class BaseConstraintModel(ABC):
    """Abstract base class for emergent constraint models"""
    
    @abstractmethod
    def fit(self, X, Y):
        """Train the model on model data"""
        pass
    
    @abstractmethod
    def get_y_c(self, X_o):
        """Predict constrained Y_c given observed X_o"""
        pass
    
    @abstractmethod
    def get_ci(self, *args: tuple):
        """Calculate 90% confidence interval"""
        pass

    @abstractmethod
    def get_model_params(self):
        """Model params resulting from the fit."""
        pass


class EMLinearModel(BaseConstraintModel):
    """Constraint model based on linear regression"""

    def __init__(self, Xbar: np.ndarray, Ybar: np.ndarray) -> None:
        self.Xbar = Xbar
        self.Ybar = Ybar
        self.model_res = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X_centred = X - self.Xbar
        Y_centred = Y - self.Ybar
        self.model_res = stats.linregress(X_centred, Y_centred)

    def get_y_c(self, Xo: np.ndarray) -> float:
        self.Yc = self.Ybar + self.model_res.slope * (Xo - self.Xbar)
        return self.Yc

    def get_ci(self, Y: np.ndarray, n: int, r: float) -> Tuple[float, float]:
        sigma = ((Y - self.Ybar) ** 2).sum() / (n - 2) * (1 - r**2)
        ci_low = self.Yc - 1.65 * sigma
        ci_high = self.Yc + 1.65 * sigma
        return sigma, ci_low, ci_high
    
    def get_model_params(self):
        return self.model_res.slope, self.model_res.intercept
    
    
EM_MODEL_TYPES = {
    "linear": EMLinearModel,
}