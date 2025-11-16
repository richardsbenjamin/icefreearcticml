from __future__ import annotations

import argparse
import json
import joblib
import os

from icefreearcticml.icefreearcticml.utils.data import read_model_data_all
from icefreearcticml.icefreearcticml.biascorrectionml.univariate import run_regression as run_uni
from icefreearcticml.icefreearcticml.biascorrectionml.multivariate import run_multivariate_bias_correction as run_multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bias correction (uni/multi) with ML regressors")
    parser.add_argument("--mode", choices=["univariate", "multivariate"], required=True)
    parser.add_argument("--variables", type=str, required=True, help="Comma-separated variable names")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--method", type=str, default="randomforest,linear,neuralnet", help="Comma-separated regressor/method names")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--params", type=str, default="{}", help="JSON of regressor params")
    parser.add_argument("--out", type=str, default="bias_correction_results.pkl")
    return parser.parse_args()

def main():
    args = parse_args()
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    params = json.loads(args.params)

    model_data = read_model_data_all()

    results = {}
    if args.mode == "multivariate":
        for model_name in models:
            for method in methods:
                results[f"{model_name}_{method}"] = run_multi(
                    model_data=model_data,
                    variables=variables,
                    model_name=model_name,
                    method=method,
                    train_split=args.train_split,
                    val_split=args.val_split,
                    **params,
                )
    else:
        for var in variables:
            for model_name in models:
                for method in methods:
                    key = f"{var}_{model_name}_{method}"
                    results[key] = run_uni(
                        model_data=model_data,
                        var=var,
                        model_name=model_name,
                        regressor_name=method,
                        regressor_params=params,
                        train_split=args.train_split,
                        val_split=args.val_split,
                    )
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"bias_correction_ml_{args.mode}.joblib")
    joblib.dump(results, out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
