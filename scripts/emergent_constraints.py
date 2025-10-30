from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats

from icefreearcticml.icefreearcticml.utils import (
    read_model_data_all,
    filter_by_years,
)
from icefreearcticml.icefreearcticml.pipeline_helpers import add_all



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


class EMLinearModel(BaseConstraintModel):
    """Constraint model based on linear regression"""

    def __init__(self, Xbar: np.ndarray, Ybar: np.ndarray) -> None:
        self.Xbar = Xbar
        self.Ybar = Ybar
        self.model_res = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X_centred = X - self.Xbar
        Y_centred = Y - self.Ybar
        self.model_res = stats.linregress(X_centred, Y_centred)  # This gives the same b

    def get_y_c(self, Xo: np.ndarray) -> float:
        self.Yc = self.Ybar + self.model_res.slope * (Xo - self.Xbar)
        return self.Yc

    def get_ci(self, Y: np.ndarray, n: int, r: float) -> Tuple[float, float]:
        """Calculate 90% confidence interval"""
        sigma = ((Y - self.Ybar) ** 2).sum() / (n - 2) * (1 - r**2)
        ci_low = self.Yc - 1.65 * sigma
        ci_high = self.Yc + 1.65 * sigma
        return ci_low, ci_high

EM_MODEL_TYPES = {
    "linear": EMLinearModel,
}


def get_em_model(model_type: str, *args: tuple) -> BaseConstraintModel:
    if model_type in EM_MODEL_TYPES:
        return EM_MODEL_TYPES[model_type](*args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def parse_args(args_list: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-varying emergent constraint with optimal historical window selection")
    p.add_argument("--var", default="ssie", help="Variable to use for both predictor (history) and target (future)")
    p.add_argument("--obs-start", type=int, default=1979)
    p.add_argument("--obs-end", type=int, default=2023)
    p.add_argument("--calib-start", type=int, default=2024)
    p.add_argument("--calib-end", type=int, default=2099)
    p.add_argument("--window", type=int, default=5, help="Projection sliding window length (years)")
    p.add_argument("--model-name", default="all", help="Name of CMPI6 model to use")
    p.add_argument("--model-type", default="linear", choices=EM_MODEL_TYPES.keys(), help="Type of constraint model")
    p.add_argument("--save-dir", default="outputs")
    if args_list is None:
        return p.parse_args()
    else:
        return p.parse_args(args_list)

def all_subperiods_np(start:int, end:int, min_len:int=5):
    years = np.arange(start, end+1)
    starts, ends = np.triu_indices(len(years), k=1)
    periods = list(zip(years[starts], years[ends]))
    return [(int(s), int(e)) for (s, e) in periods if (e - s + 1) >= min_len]

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
        r = float(np.corrcoef(X, Y)[0, 1])
        if np.isnan(r):
            continue
        if r > best_r:
            best_r = r
            best_win = (hs, he)

    return best_win, best_r

def run_time_varying_ec(model_data: Dict, args: argparse.Namespace) -> pd.DataFrame:
    hist_df: DataFrame = model_data[args.hist_var][args.model_name]
    fut_df: DataFrame = model_data[args.fut_var][args.model_name]
    obs_series: DataFrame = model_data[args.var]["Observations"]
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

        ci_low, ci_high = model.get_ci(Y, n, r)
        rows.append({
            "proj_start": ps,
            "proj_end": pe,
            "hist_start": hs,
            "hist_end": he,
            "r": r,
            "raw_mean": Ybar,
            "Yc": Yc,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
        return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model_data: Dict = read_model_data_all()
    add_all(model_data)

    df = run_time_varying_ec(model_data, args)

    # save both csv and joblib
    jb_path = os.path.join(args.save_dir, f"emergent_constraints_{args.hist_var}_{args.fut_var}_{args.model_type}.joblib")
    joblib.dump({"config": args, "results": df}, jb_path)
    print(f"Saved:\n  {jb_path}")


if __name__ == "__main__":
    main()