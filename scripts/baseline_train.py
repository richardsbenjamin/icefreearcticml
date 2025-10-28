from __future__ import annotations

import argparse
import os
import joblib
from typing import List

from icefreearcticml.tft_helpers import (
    TrainConfig,
    run_model_from_config,
)
from icefreearcticml.utils import read_model_data_all
from icefreearcticml.pipeline_helpers import add_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline training (TemporalFusionTransformer)")
    p.add_argument("--model-name", default="all", help="Model name to train on (default: all)")
    p.add_argument("--train-split", type=float, default=0.8, help="Train split fraction (default: 0.8)")
    p.add_argument("--max-encoder-length", type=int, default=10, help="Max encoder length")
    p.add_argument("--max-prediction-length", type=int, default=1, help="Max prediction length")
    p.add_argument("--y-var", default="ssie", help="Target variable (default: ssie)")
    p.add_argument(
        "--x-vars",
        default="tas,wsiv,oht_atl,oht_pac",
        help="Comma-separated input variables",
    )
    p.add_argument("--save-dir", default=True, help="Optional directory to save outputs (joblib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_vars: List[str] = [v for v in args.x_vars.split(",") if v]

    # Load data
    model_data = read_model_data_all()
    add_all(model_data)

    # Configure training
    train_config = TrainConfig(
        y_var=args.y_var,
        x_vars=x_vars,
        train_split=args.train_split,
        model_name=args.model_name,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
    )

    # Train and get outputs
    output = run_model_from_config(train_config, model_data)

    # Optionally save outputs
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "baseline_output.joblib")
        joblib.dump(output, out_path)
        print(f"Saved baseline output -> {out_path}")
    else:
        # Print brief summary
        print("Baseline train completed.")
        print("Config:", train_config.__dict__)


if __name__ == "__main__":
    main()