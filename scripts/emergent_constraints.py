import argparse
import os

import joblib

from icefreearcticml.icefreearcticml.emergent_constraints import (
    EM_MODEL_TYPES,
    run_time_varying_ec,
)
from icefreearcticml.icefreearcticml.utils.data import (
    read_model_data_all,
)

def parse_args(args_list: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-varying emergent constraint with optimal historical window selection")
    p.add_argument("--hist-var", default="tas", help="Comma separated variables to use for historical window selection")
    p.add_argument("--fut-var", default="ssie", help="Variable to use for future window selection")
    p.add_argument("--obs-start", type=int, default=1979)
    p.add_argument("--obs-end", type=int, default=2023)
    p.add_argument("--calib-start", type=int, default=2024)
    p.add_argument("--calib-end", type=int, default=2099)
    p.add_argument("--window", type=int, default=5, help="Projection sliding window length (years)")
    p.add_argument("--model-type", default="linear", choices=EM_MODEL_TYPES.keys(), help="Type of constraint model")
    p.add_argument("--save-dir", default="outputs")
    if args_list is None:
        return p.parse_args()
    else:
        return p.parse_args(args_list)

def main(args_list: list | None = None) -> None:
    args = parse_args(args_list)
    os.makedirs(args.save_dir, exist_ok=True)

    hist_vars = [v.strip() for v in args.hist_var.split(",") if v.strip()]
    model_data = read_model_data_all()

    for hist_var in hist_vars:
        df = run_time_varying_ec(model_data, hist_var, args)
        # save both csv and joblib
        jb_path = os.path.join(
            args.save_dir,
            f"emergent_constraints_{hist_var}_{args.fut_var}_{args.model_type}.joblib",
        )
        joblib.dump({"config": args, "results": df}, jb_path)
        print(f"Saved:\n  {jb_path}")


if __name__ == "__main__":
    main()