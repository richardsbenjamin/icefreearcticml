import argparse
import os
from typing import List

import joblib
from pandas import DataFrame

from icefreearcticml.icefreearcticml.learningdistribution import detrend_with_trend, run_model_from_config
from icefreearcticml.icefreearcticml.utils.data import read_model_data_all
from icefreearcticml.icefreearcticml.utils.trainconfig import TrainConfig
from icefreearcticml.icefreearcticml.utils.utils import get_train_test_ensembles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distribution learning with Temporal Fusion Transformer")
    p.add_argument("--model-name", default="all", help="Model name to train on (default: all)")
    p.add_argument("--train-split", type=float, default=0.8, help="Train split fraction (default: 0.8)")
    p.add_argument("--max-encoder-length", type=int, default=10, help="Max encoder length")
    p.add_argument("--max-prediction-length", type=int, default=76, help="Max prediction length")
    p.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    p.add_argument(
        "--vars",
        required=True,
        help="Comma-separated variables for both target and input",
    )
    p.add_argument("--save-dir", required=True, help="Directory to save outputs (joblib)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    vars_: List[str] = [v for v in args.vars.split(",") if v]

    model_data = read_model_data_all()

    members = model_data[vars_[0]][args.model_name].columns
    n_members = len(members)
    train_split = 0.8
    train_members_int, test_members_int = get_train_test_ensembles(n_members, train_split)
    train_members, test_members = members[train_members_int], members[test_members_int]

    # Detrend data
    model_data_detrended = {}
    trends = {}
    for var in vars_:
        df = model_data[var][args.model_name].fillna(0)
        residuals_df, pred_df = detrend_with_trend(df, train_members)
        model_data_detrended[var] = {args.model_name: DataFrame(residuals_df, index=df.index)}
        trends[var] = pred_df

    train_config = TrainConfig(
        y_var=vars_,
        x_vars=vars_,
        train_split=args.train_split,
        model_name=args.model_name, 
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        epochs=args.epochs,
    )
    train_config.set_prior_train_test_members(train_members, test_members)
    model_res = run_model_from_config(train_config, model_data_detrended)
    model_res["trends"] = trends
    model_res["config"] = args

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"distribution_learning_{args.model_name}.joblib")
    joblib.dump(model_res, out_path)
    print(f"Saved output -> {out_path}")


if __name__ == "__main__":
    main()