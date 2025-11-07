from __future__ import annotations

import argparse
import os
from typing import Dict, List, Any

import joblib
import xarray as xr
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

from icefreearcticml.icefreearcticml.utils import (
    read_model_data_all,
    calculate_first_icefree_year,
)
from icefreearcticml.icefreearcticml.pipeline_helpers import (
    add_all,
    get_y_emulated_outputs,
    get_y_simulated_outputs,
)
from icefreearcticml.icefreearcticml.pipeline_helpers import (
    LiangConfig,
    calculate_all_liang_flows,
)
from icefreearcticml.icefreearcticml.pipeline_helpers import get_liangs_from_output_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate experiments: RMSE, Liang index, emulated ice-free years")
    p.add_argument("--jobs", required=True, help="Comma-separated list of joblib files (each an experiment)")
    p.add_argument("--ice-ds", required=True, help="Path to outputs/ice_free_year_ds.nc from precomp")
    p.add_argument("--liang-start", default="1980-01-01")
    p.add_argument("--liang-end", default="2060-01-01")
    p.add_argument("--x-liang", default="wsiv,tas,oht_atl,oht_pac")
    p.add_argument("--y-liang", default="ssie")
    p.add_argument("--save-dir", default="outputs")
    return p.parse_args()

def flatten_outputs(obj: Any) -> Dict[str, Any]:
    """Flatten nested dicts to a flat mapping of labels -> Output-like objects."""
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = flatten_outputs(v)
            if sub:
                for sk, sv in sub.items():
                    flat[f"{k}/{sk}"] = sv
            else:
                flat[k] = v
    else:
        # leaf
        return {}
    return flat

def compute_rmse_for_output(output, model_data: Dict) -> float:
    """RMSE between emulated and simulated test series for y_var, aligned on time and members."""
    y_pred: DataFrame = get_y_emulated_outputs(output)  # time x members
    y_true: DataFrame = get_y_simulated_outputs(output, model_data)
    # Align indices and columns
    common_index = y_pred.index.intersection(y_true.index)
    common_cols = y_pred.columns.intersection(y_true.columns)
    y_pred_a = y_pred.loc[common_index, common_cols]
    y_true_a = y_true.loc[common_index, common_cols]
    return float(mean_squared_error(y_true_a.values.ravel(), y_pred_a.values.ravel(), squared=False))

def compute_emulated_ify_for_output(output) -> DataFrame:
    """Compute emulated ice-free years per member from emulated series."""
    y_pred: DataFrame = get_y_emulated_outputs(output)
    # calculate_first_icefree_year expects DataFrame with time index and member cols
    years = calculate_first_icefree_year(y_pred)
    return DataFrame({"member": y_pred.columns, "ify_emulated": years})

def compute_original_ify_subset(ice_ds: xr.Dataset, members: List[str]) -> DataFrame:
    sel = ice_ds.sel(ensemble=members)
    return DataFrame({"member": members, "ify_original": sel["ice_free_years"].values})

def build_liang_config(args: argparse.Namespace, x_vars: List[str], y_var: str) -> LiangConfig:
    return LiangConfig(args.liang_start, args.liang_end, x_vars, y_var, dt=1, n_iter=1000)

def compute_liang_for_original(model_data: Dict, members: List[str], x_vars: List[str], y_var: str, liang_config: LiangConfig):
    # Filter model_data to members
    filtered = {}
    for var in [*x_vars, y_var]:
        filtered[var] = {"all": model_data[var]["all"][members]}
    return calculate_all_liang_flows(filtered, liang_config, model_names=["all"])

def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    x_liang = [v for v in args.x_liang.split(",") if v]
    y_liang = args.y_liang

    # Data
    model_data = read_model_data_all()
    add_all(model_data)
    ice_ds = xr.load_dataset(args.ice_ds)

    liang_config = build_liang_config(args, x_liang, y_liang)

    job_paths = [p for p in args.jobs.split(",") if p]

    summary_rows = []
    all_liang_results = {}
    all_ify_compare = {}

    for job_path in job_paths:
        exp_obj = joblib.load(job_path)
        outputs_map = flatten_outputs(exp_obj) if isinstance(exp_obj, dict) else {"root": exp_obj}

        exp_name = os.path.splitext(os.path.basename(job_path))[0]
        all_liang_results[exp_name] = {}
        all_ify_compare[exp_name] = {}

        for key, output in outputs_map.items():
            # test members for fair comparison
            test_members = list(output.train_config.test_members)

            # RMSE
            rmse = compute_rmse_for_output(output, model_data)

            # Emulated IFY vs original IFY on test members
            ify_emul_df = compute_emulated_ify_for_output(output)
            ify_emul_df = ify_emul_df[ify_emul_df["member"].isin(test_members)].set_index("member").sort_index()
            ify_orig_df = compute_original_ify_subset(ice_ds, test_members).set_index("member").sort_index()
            years_rmse = float(mean_squared_error(ify_orig_df.values.ravel(), ify_emul_df.values.ravel(), squared=False))
            all_ify_compare[exp_name][key] = {
                "emulated": ify_emul_df["ify_emulated"].to_dict(),
                "original": ify_orig_df["ify_original"].to_dict(),
                "rmse_years": years_rmse,
            }

            # Liang: original and emulated for this output
            original_liang = compute_liang_for_original(model_data, test_members, x_liang, y_liang, liang_config)
            emul_liang = get_liangs_from_output_dict({key: output}, liang_config, model_names=["all"])  # uses helper
            all_liang_results[exp_name][key] = {
                "original": original_liang,
                "emulated": emul_liang,
            }

            summary_rows.append({
                "experiment": exp_name,
                "key": key,
                "rmse": rmse,
                "rmse_years": years_rmse,
            })

    # Save artifacts
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.save_dir, "evaluation_summary.csv")
    joblib.dump(all_liang_results, os.path.join(args.save_dir, "evaluation_liang.joblib"))
    joblib.dump(all_ify_compare, os.path.join(args.save_dir, "evaluation_ify.joblib"))
    summary_df.to_csv(summary_csv, index=False)

    print("Saved:")
    print(f"  {summary_csv}")
    print(f"  {os.path.join(args.save_dir, 'evaluation_liang.joblib')}")
    print(f"  {os.path.join(args.save_dir, 'evaluation_ify.joblib')}")


if __name__ == "__main__":
    main()