#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/bias_correction_ml.out"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

UNIVAR_METHODS="random_forest,linear,gradient_boosting,neural_network"
MULTIVAR_METHODS="random_forest,linear,neural_network,chained_rf"

MODE_ARG="multivariate" # univariate or multivariate
VARS_ARG="ssie,tas,wsiv,oht_atl,oht_pac"
MODELS_ARG="EC-Earth3,CESM2,MPI-ESM1-2-LR,CanESM5,ACCESS-ESM1-5,all"
METHOD_ARG=${MULTIVAR_METHODS}
PARAMS_ARG="{}"
TRAIN_SPLIT_ARG=0.8
VAL_SPLIT_ARG=0.1

RUN_CMD="python scripts/bias_correction_ml.py \
    --mode ${MODE_ARG} \
    --variables ${VARS_ARG} \
    --models ${MODELS_ARG} \
    --method ${METHOD_ARG} \
    --train-split ${TRAIN_SPLIT_ARG} \
    --val-split ${VAL_SPLIT_ARG} \
    --params ${PARAMS_ARG} \
    --out ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
