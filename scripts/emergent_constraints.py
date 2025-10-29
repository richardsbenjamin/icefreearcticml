from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from icefreearcticml.utils import (
    read_model_data_all,
    filter_by_years,
)
from icefreearcticml.pipeline_helpers import add_all


@dataclass
class ECConfig:
    predictor_var: str = "tas"
    target_var: str = "ssie"
    hist_start: int = 1980
    hist_end: int = 2014
    fut_start: int = 2030
    fut_end: int = 2060
    time_varying: bool = False
    method: str = "linear"     # "linear" | "mlp"
    target_stat: str = "mean"  # "mean" | "trend"
    model_name: str = "all"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run emergent constraints (linear or MLP; static or time-varying)")
    p.add_argument("--predictor-var", default="tas")
    p.add_argument("--target-var", default="ssie")
    p.add_argument("--hist-start", type=int, default=1980)
    p.add_argument("--hist-end", type=int, default=2014)
    p.add_argument("--fut-start", type=int, default=2030)
    p.add_argument("--fut-end", type=int, default=2060)
    p.add_argument("--time-varying", action="store_true", help="Use time-varying EC (sequence features)")
    p.add_argument("--method", choices=["linear", "mlp"], default="linear")
    p.add_argument("--target-stat", choices=["mean", "trend"], default="mean")
    p.add_argument("--model-name", default="all", help="Which model set in model_data to use")
    p.add_argument("--save-dir", default="outputs")
    return p.parse_args()


def get_member_matrix(df: DataFrame, start_year: int, end_year: int) -> np.ndarray:
    sub = filter_by_years(df, start_year, end_year)
    # rows: years, cols: members
    return sub.to_numpy()


def get_predictor_features(df: DataFrame, cfg: ECConfig) -> Tuple[np.ndarray, StandardScaler | None]:
    mat = get_member_matrix(df, cfg.hist_start, cfg.hist_end)
    if not cfg.time_varying:
        # feature: mean over historical period per member
        feats = mat.mean(axis=0).reshape(-1, 1)
        return feats, None
    # time-varying: use entire historical sequence (year dimension)
    # shape (members, time)
    feats = mat.T
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    return feats, scaler


def linear_trend(y: np.ndarray) -> float:
    x = np.arange(len(y)).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    return float(reg.coef_[0])


def get_target_values(df: DataFrame, cfg: ECConfig) -> np.ndarray:
    mat = get_member_matrix(df, cfg.fut_start, cfg.fut_end)
    if cfg.target_stat == "mean":
        return mat.mean(axis=0)
    elif cfg.target_stat == "trend":
        # compute per-member linear trend over future window
        return np.apply_along_axis(linear_trend, 0, mat)
    raise ValueError("Unknown target_stat")


def fit_model(X: np.ndarray, y: np.ndarray, method: str):
    if method == "linear":
        model = LinearRegression()
    else:
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", random_state=42, max_iter=2000)
    model.fit(X, y)
    return model


def constrain_projection(model, X_obs: np.ndarray) -> float:
    return float(model.predict(X_obs.reshape(1, -1))[0])


def build_observed_feature(obs_series: DataFrame, cfg: ECConfig, scaler: StandardScaler | None) -> np.ndarray:
    obs_hist = filter_by_years(obs_series, cfg.hist_start, cfg.hist_end).to_numpy().reshape(-1)
    if not cfg.time_varying:
        return np.array([obs_hist.mean()])
    vec = obs_hist.reshape(1, -1)
    if scaler is not None:
        vec = scaler.transform(vec)
    return vec.reshape(-1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = ECConfig(
        predictor_var=args.predictor_var,
        target_var=args.target_var,
        hist_start=args.hist_start,
        hist_end=args.hist_end,
        fut_start=args.fut_start,
        fut_end=args.fut_end,
        time_varying=args.time_varying,
        method=args.method,
        target_stat=args.target_stat,
        model_name=args.model_name,
    )

    # Load data
    model_data: Dict = read_model_data_all()
    add_all(model_data)

    # Prepare predictor features (members x features)
    pred_df: DataFrame = model_data[cfg.predictor_var][cfg.model_name]
    X, scaler = get_predictor_features(pred_df, cfg)

    # Prepare target values (members,)
    targ_df: DataFrame = model_data[cfg.target_var][cfg.model_name]
    y = get_target_values(targ_df, cfg)

    # Fit model
    model = fit_model(X, y, cfg.method)
    r2 = float(model.score(X, y))

    # Observed constraint
    obs_series: DataFrame = model_data[cfg.predictor_var]["Observations"]
    x_obs = build_observed_feature(obs_series, cfg, scaler)
    constrained = constrain_projection(model, x_obs)

    # Unconstrained mean (raw multi-member)
    raw_mean = float(y.mean())
    delta = constrained - raw_mean

    # Save results
    out = {
        "config": cfg.__dict__,
        "r2": r2,
        "raw_mean": raw_mean,
        "constrained": constrained,
        "delta": delta,
        "n_members": int(X.shape[0]),
    }
    out_path = os.path.join(args.save_dir, "emergent_constraints.joblib")
    joblib.dump(out, out_path)
    print(f"Saved EC results -> {out_path}")


if __name__ == "__main__":
    main()