from __future__ import annotations

import argparse
import json
import joblib

from icefreearcticml.taylortransformer.utils import prepare_data, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Taylor Transformer bias correction (uni/multi)")
    parser.add_argument("--mode", choices=["univariate", "multivariate"], required=True)
    parser.add_argument("--variables", type=str, required=True, help="Comma-separated variable names")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--config", type=str, default="{}", help="JSON for training config (batch_size, min_context, etc.)")
    parser.add_argument("--out", type=str, default="taylortransformer_bc_results.pkl")
    parser.add_argument("--tag-prefix", type=str, default="taylor", help="Prefix for saved checkpoints model_tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    config = json.loads(args.config)

    results = {}

    if args.mode == "univariate":
        for var in variables:
            for model_name in models:
                data_res = prepare_data(variables=[var], model_name=model_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
                tag = f"{args.tag_prefix}_{model_name}_{var}"
                results[f"{var}_{model_name}"] = train_model(data_res, config, epochs=args.epochs, model_tag=tag)
    else:
        for model_name in models:
            data_res = prepare_data(variables=variables, model_name=model_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
            tag = f"{args.tag_prefix}_{model_name}"
            results[model_name] = train_model(data_res, config, epochs=args.epochs, model_tag=tag)

    joblib.dump(results, args.out)
    print(f"Saved TT bias correction results to {args.out}")


if __name__ == "__main__":
    main()