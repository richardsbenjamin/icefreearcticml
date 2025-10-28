from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Dict, List

import joblib
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from icefreearcticml.tft_helpers import TrainConfig, run_model_from_config
from icefreearcticml.tft_helpers import get_members_by_percentile
from icefreearcticml.utils import read_model_data_all
from icefreearcticml.pipeline_helpers import add_all


# ----------------------
# CLI
# ----------------------

METHOD_CHOICES = [
    "abs_large_remove",
    "warm_cold_large_remove",
    "abs_small_remove",
    "warm_cold_small_remove",
    "abs_half_split",
    "warm_cold_half_split",
    "abs_cluster",
    "warm_cold_cluster",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias removal training experiments")
    p.add_argument("--method", required=True, choices=METHOD_CHOICES, help="Experiment method to run")
    p.add_argument("--bias-ds", default="outputs/bias_ds.nc", help="Path to bias_ds netcdf output from precomp")
    p.add_argument("--bias-vars", default="tas,wsiv,oht_atl,oht_pac", help="Comma-separated bias variable base names")
    p.add_argument("--percentiles", default="0.01,0.02,0.05", help="Comma-separated percentiles (e.g., 0.01,0.02)")
    p.add_argument("--model-name", default="all", help="Model name to train on (default: all)")
    p.add_argument("--train-split", type=float, default=0.8, help="Train split fraction")
    p.add_argument("--y-var", default="ssie", help="Target variable")
    p.add_argument("--x-vars", default="tas,wsiv,oht_atl,oht_pac", help="Comma-separated input variables for training")
    p.add_argument("--max-encoder-length", type=int, default=10)
    p.add_argument("--max-prediction-length", type=int, default=1)
    p.add_argument("--save-dir", default="outputs", help="Directory to save joblib outputs")
    return p.parse_args()


# ----------------------
# Core helpers
# ----------------------

def load_bias_ds(path: str) -> xr.Dataset:
    return xr.load_dataset(path)


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    x_vars: List[str] = [v for v in args.x_vars.split(",") if v]
    return TrainConfig(
        y_var=args.y_var,
        x_vars=x_vars,
        train_split=args.train_split,
        model_name=args.model_name,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
    )


def run_with_member_filter(
    base_config: TrainConfig,
    model_data: Dict,
    members,
):
    cfg = deepcopy(base_config)
    cfg.set_members_for_model(list(members))
    return run_model_from_config(cfg, model_data)


def run_percentile_experiment(
    bias_ds: xr.Dataset,
    base_config: TrainConfig,
    model_data: Dict,
    bias_vars: List[str],
    percentiles: List[float],
    var_name_suffix: str,
    filter_type: str,
) -> Dict[str, Dict[float, object]]:
    results: Dict[str, Dict[float, object]] = {}
    for bias_var in bias_vars:
        var_name = f"{bias_var}_{var_name_suffix}"
        outputs: Dict[float, object] = {}
        for p in percentiles:
            p_in = (1 - p) if filter_type == "lt_inverted" else p
            ft = "lt" if filter_type in ("lt", "lt_inverted") else "gte"
            members = get_members_by_percentile(bias_ds, var_name, p_in, type_=ft)
            outputs[p] = run_with_member_filter(base_config, model_data, members)
        results[bias_var] = outputs
    return results


def run_half_split(
    bias_ds: xr.Dataset,
    base_config: TrainConfig,
    model_data: Dict,
    bias_vars: List[str],
    var_name_suffix: str,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for bias_var in bias_vars:
        var_name = f"{bias_var}_{var_name_suffix}"
        outputs: Dict[str, object] = {}
        for ft in ("lt", "gte"):
            members = get_members_by_percentile(bias_ds, var_name, 0.5, type_=ft)
            outputs[ft] = run_with_member_filter(base_config, model_data, members)
        results[bias_var] = outputs
    return results


def run_cluster_experiment(
    bias_ds: xr.Dataset,
    base_config: TrainConfig,
    model_data: Dict,
    bias_vars: List[str],
    var_name_suffix: str,
) -> Dict[str, Dict[int, object]]:
    results: Dict[str, Dict[int, object]] = {}
    for bias_var in bias_vars:
        var_name = f"{bias_var}_{var_name_suffix}"
        X = bias_ds[var_name].values.reshape(-1, 1)
        # choose best k by silhouette
        best_k, best_score = None, -1
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(X)
        clusters_da = xr.DataArray(labels, dims=["ensemble"], coords={"ensemble": bias_ds["ensemble"].values})

        outputs: Dict[int, object] = {}
        for k in range(best_k):
            members = clusters_da.where(clusters_da == k, drop=True)["ensemble"].values
            outputs[k] = run_with_member_filter(base_config, model_data, members)
        results[bias_var] = outputs
    return results


def get_save_name(method: str) -> str:
    return f"{method}_outputs.joblib"


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    bias_vars: List[str] = [v for v in args.bias_vars.split(",") if v]
    percentiles: List[float] = [float(x) for x in args.percentiles.split(",") if x]

    # load data
    model_data = read_model_data_all()
    add_all(model_data)
    bias_ds = load_bias_ds(args.bias_ds)

    base_config = build_train_config(args)

    # dispatch by method
    if args.method == "abs_large_remove":
        res = run_percentile_experiment(bias_ds, base_config, model_data, bias_vars, percentiles, "bias_abs", "lt_inverted")
    elif args.method == "warm_cold_large_remove":
        res = run_percentile_experiment(bias_ds, base_config, model_data, bias_vars, percentiles, "bias_abs", "lt_inverted")
    elif args.method == "abs_small_remove":
        res = run_percentile_experiment(bias_ds, base_config, model_data, bias_vars, percentiles, "bias_abs", "gte")
    elif args.method == "warm_cold_small_remove":
        res = run_percentile_experiment(bias_ds, base_config, model_data, bias_vars, percentiles, "bias", "gte")
    elif args.method == "abs_half_split":
        res = run_half_split(bias_ds, base_config, model_data, bias_vars, "bias_abs")
    elif args.method == "warm_cold_half_split":
        res = run_half_split(bias_ds, base_config, model_data, bias_vars, "bias")
    elif args.method == "abs_cluster":
        res = run_cluster_experiment(bias_ds, base_config, model_data, bias_vars, "bias_abs")
    elif args.method == "warm_cold_cluster":
        res = run_cluster_experiment(bias_ds, base_config, model_data, bias_vars, "bias")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    out_path = os.path.join(args.save_dir, get_save_name(args.method))
    joblib.dump(res, out_path)
    print(f"Saved experiment outputs -> {out_path}")


if __name__ == "__main__":
    main()