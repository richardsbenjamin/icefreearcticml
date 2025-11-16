# Refining CMIP6 projections of an ice-free Arctic using machine learning

Code repository for Master's thesis "Refining CMIP6 projections of an ice-free Arctic using machine learning".

## Setting up your environment

To set up the repo and required Python environment:

1.  Clone the repository:
    ```sh
    git clone https://github.com/richardsbenjamin/icefreearcticml.git
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Data Directory

All expected input data should be placed in the `data/` folder with the following structure:

```data/
├── Timeseries_ACCESS-ESM1-5.npy
├── Timeseries_CanESM5.npy
├── Timeseries_MPI-ESM1-2-LR.npy
├── Timeseries_CESM2.npy
├── Timeseries_EC-Earth3.npy
└── Timeseries_Observations.npy
```

> **Note:** The `data/` directory is intentionally left empty in version control. The required CMIP6 model output files are not publicly distributed with this repository. To obtain access to these data files, please contact the author directly.

## Scripts
The analysis pipeline consists of several scripts that should be run in sequence. Some scripts need to be run multiple times for different model configurations and variables.

```
# Precomputation of member biases and ice-free years in CMIP6 data
./scripts/precomp.sh

# Statistical bias correction methods (linear scaling, variance scaling, quantile mapping)
./scripts/bias_correction.sh

# Time-varying emergent constraints analysis
./scripts/emergent_constraints.sh

# TemporalFusionTransformer baseline model training
./scripts/baseline_train.sh

# Bias experiments using the TemporalFusionTransformer architecture
./scripts/bias_removal_train.sh

# Bias correction using the TaylorFormer transformer model
./scripts/bias_correction_taylortransform.sh

# Learning distribution shifts in CMIP6 data using TemporalFusionTransformer
./scripts/learningdistribution.sh
```

## Results
After running all necessary scripts, the results can be inspected and visualised using the `Results.ipynb` Jupyter notebook. For the imports to work properly, place the notebook in the root directory of the repository as shown below:
```
root/
├── icefreearcticml/
│   ├── ...
│   └── README.md
└── Results.ipynb
```