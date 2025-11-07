from __future__ import annotations

import argparse
import os
import joblib
from typing import Dict, List

import xarray as xr  # for type hints/optional saves if needed

from cmethods import adjust  # noqa: F401 (kept for completeness if needed downstream)
from cmethods.core import apply_ufunc as apply_correction

from icefreearcticml.constants import VARIABLES as VAR_NAMES, MODELS
from icefreearcticml.utils import read_model_data_all
from icefreearcticml.pipeline_utils import add_all
from icefreearcticml.bias_correction_utils import (
    DEFAULT_BIAS_METHODS,
    DEFAULT_MODEL_NAMES,
    compute_bias_corrections,
    score_bias_corrections,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run bias corrections and scoring")
    p.add_argument("--vars", default=",".join(VAR_NAMES), help="Comma-separated variables to correct")
    p.add_argument("--methods", default=",".join(DEFAULT_BIAS_METHODS.keys()), help="Comma-separated correction methods to use")
    p.add_argument("--save-dir", default="outputs", help="Directory to save outputs")
    return p.parse_args()


def select_methods(methods_csv: str) -> Dict[str, dict]:
    names = [m for m in methods_csv.split(",") if m]
    selected = {k: v for k, v in DEFAULT_BIAS_METHODS.items() if k in names}
    if not selected:
        raise ValueError("No valid methods selected. Check --methods against available methods.")
    return selected


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    variables: List[str] = [v for v in args.vars.split(",") if v]
    methods: Dict[str, dict] = select_methods(args.methods)

    # Load and prepare model data
    model_data: Dict = read_model_data_all()
    add_all(model_data)

    # Compute corrections
    bias_corrections = compute_bias_corrections(
        model_data=model_data,
        bias_methods=methods,
        var_names=variables,
        model_names=None,  # defaults to models excluding Observations
        apply_correction=apply_correction,
    )

    # Score corrections
    scores_dict, mse_df = score_bias_corrections(
        bias_corrections=bias_corrections,
        model_data=model_data,
        bias_vars=variables,
        model_names=None,
    )

    # Save outputs
    corrections_path = os.path.join(args.save_dir, "bias_corrections.joblib")
    scores_path = os.path.join(args.save_dir, "bias_correction_scores.joblib")
    mse_csv_path = os.path.join(args.save_dir, "bias_correction_mse.csv")

    joblib.dump(bias_corrections, corrections_path)
    joblib.dump(scores_dict, scores_path)
    mse_df.to_csv(mse_csv_path)

    print("Saved:")
    print(f"  {corrections_path}")
    print(f"  {scores_path}")
    print(f"  {mse_csv_path}")


if __name__ == "__main__":
    main()
