#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/bias_correction_taylortransform.out"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

# Defaults (can be overridden by environment variables before calling this script)
MODE_ARG="multivariate"             # univariate | multivariate
VARS_ARG="ssie,tas,wsiv,oht_atl,oht_pac"
MODELS_ARG="EC-Earth3,CESM2,MPI-ESM1-2-LR,CanESM5,ACCESS-ESM1-5,all"
TRAIN_RATIO_ARG="0.8"
VAL_RATIO_ARG="0.1"
MIN_CONTEXT="2"
MAX_CONTEXT="43"
BATCH_SIZE="32"
EPOCHS="5000"
ACCUM_STEPS="500" # Accumulation steps for the LR scheduler
TOTAL_LENGTH="45" # Total window (n_C + n_T)
WARMUP_RATIO="0.1"
OUT_ARG="${OUTPUT_DIR}/taylortransformer_bc_results.pkl"
TAG_PREFIX_ARG="taylor"

RUN_CMD="python scripts/bias_correction_taylortransform.py \
    --mode ${MODE_ARG} \
    --variables ${VARS_ARG} \
    --models ${MODELS_ARG} \
    --train-ratio ${TRAIN_RATIO_ARG} \
    --val-ratio ${VAL_RATIO_ARG} \
    --total-length ${TOTAL_LENGTH} \
    --warmup-ratio ${WARMUP_RATIO} \
    --accumulation-steps ${ACCUM_STEPS} \
    --epochs ${EPOCHS} \
    --min-context ${MIN_CONTEXT} \
    --max-context ${MAX_CONTEXT} \
    --batch-size ${BATCH_SIZE} \
    --out ${OUT_ARG} \
    --tag-prefix ${TAG_PREFIX_ARG}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
