from __future__ import annotations

import argparse
import json
import joblib

from icefreearcticml.icefreearcticml.taylortransformer.utils import prepare_data, train_model
from icefreearcticml.icefreearcticml.utils.data import read_model_data_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Taylor Transformer bias correction (uni/multi)")
    parser.add_argument("--mode", choices=["univariate", "multivariate"], required=True)
    parser.add_argument("--variables", type=str, required=True, help="Comma-separated variable names")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--total-length", type=int, help="Total window length (n_C + n_T) for Taylor Transformer")
    parser.add_argument("--warmup-ratio", type=float, help="Ratio of training steps for learning rate warmup")
    parser.add_argument("--accumulation-steps", type=int, help="Gradient accumulation steps for effective batch size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--min-context", type=int, help="Minimum context length for training sequences")
    parser.add_argument("--max-context", type=int, help="Maximum context length for training sequences")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--out", type=str, default="taylortransformer_bc_results.pkl")
    parser.add_argument("--tag-prefix", type=str, default="taylor", help="Prefix for saved checkpoints model_tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    model_data = read_model_data_all()

    results = {}

    if args.mode == "univariate":
        for var in variables:
            for model_name in models:
                data_res = prepare_data(model_data, variables=[var], model_name=model_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
                tag = f"{args.tag_prefix}_{model_name}_{var}"
                results[f"{var}_{model_name}"] = train_model(data_res, args, model_tag=tag)
    else:
        for model_name in models:
            data_res = prepare_data(model_data, variables=variables, model_name=model_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
            tag = f"{args.tag_prefix}_{model_name}"
            results[model_name] = train_model(data_res, args, model_tag=tag)

    joblib.dump(results, args.out)
    print(f"Saved TT bias correction results to {args.out}")


if __name__ == "__main__":
    main()