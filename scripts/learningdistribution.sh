#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/distribution_learning.out"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

VARS_ARG="ssie,tas,wsiv,oht_atl,oht_pac"
MODEL_NAME_ARG="all"
TRAIN_SPLIT_ARG="0.8"
MAX_ENCODER_ARG="10"
MAX_PRED_ARG="76"
EPOCHS_ARG="15"

RUN_CMD="python scripts/learningdistribution.py \
    --model-name ${MODEL_NAME_ARG} \
    --train-split ${TRAIN_SPLIT_ARG} \
    --max-encoder-length ${MAX_ENCODER_ARG} \
    --max-prediction-length ${MAX_PRED_ARG} \
    --epochs ${EPOCHS_ARG} \
    --vars ${VARS_ARG} \
    --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
