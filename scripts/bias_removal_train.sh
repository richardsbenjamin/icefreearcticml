#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/bias_removal_train.out"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

# Optional environment overrides
METHOD_ARG="warm_cold_large_remove"
BIAS_DS_ARG="${OUTPUT_DIR}/bias_ds.nc"
BIAS_VARS_ARG="tas" # wsiv,oht_atl,oht_pac
PERCENTILES_ARG="0.05,0.1,0.2"

MODEL_NAME_ARG="all"
TRAIN_SPLIT_ARG="0.8"
MAX_ENCODER_ARG="10"
MAX_PRED_ARG="1"
Y_VAR_ARG="ssie"
X_VARS_ARG="tas" # tas,wsiv,oht_atl,oht_pac

RUN_CMD="python scripts/bias_removal_train.py \
    --method ${METHOD_ARG} \
    --bias-ds ${BIAS_DS_ARG} \
    --bias-vars ${BIAS_VARS_ARG} \
    --percentiles ${PERCENTILES_ARG} \
    --model-name ${MODEL_NAME_ARG} \
    --train-split ${TRAIN_SPLIT_ARG} \
    --max-encoder-length ${MAX_ENCODER_ARG} \
    --max-prediction-length ${MAX_PRED_ARG} \
    --y-var ${Y_VAR_ARG} \
    --x-vars ${X_VARS_ARG} \
    --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
