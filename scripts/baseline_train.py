from __future__ import annotations

import argparse
import os
import joblib
from typing import List

from icefreearcticml.icefreearcticml.tft import run_model_from_config
from icefreearcticml.icefreearcticml.utils.data import read_model_data_all
from icefreearcticml.icefreearcticml.utils.trainconfig import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline training (TemporalFusionTransformer)")
    p.add_argument("--model-name", default="all", help="Comma separated model name to train on (default: all)")
    p.add_argument("--train-split", type=float, default=0.8, help="Train split fraction (default: 0.8)")
    p.add_argument("--max-encoder-length", type=int, default=10, help="Max encoder length")
    p.add_argument("--max-prediction-length", type=int, default=1, help="Max prediction length")
    p.add_argument("--y-var", default="ssie", help="Target variable (default: ssie)")
    p.add_argument(
        "--x-vars",
        default="tas,wsiv,oht_atl,oht_pac",
        help="Comma-separated input variables",
    )
    p.add_argument("--save-dir", help="Directory to save outputs (joblib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_names: List[str] = [v for v in args.model_name.split(",") if v]
    x_vars: List[str] = [v for v in args.x_vars.split(",") if v]

    model_data = read_model_data_all()

    for model_name in model_names:
        train_config = TrainConfig(
            y_var=args.y_var,
            x_vars=x_vars,
            train_split=args.train_split,
            model_name=model_name,
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
        )

        output = run_model_from_config(train_config, model_data)

        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, f"tft_baseline_output_{model_name}.joblib")
        joblib.dump(output, out_path)
        print(f"Saved baseline output -> {out_path}")
        print("Baseline train completed.")
        print("Config:", train_config.__dict__)


if __name__ == "__main__":
    main()