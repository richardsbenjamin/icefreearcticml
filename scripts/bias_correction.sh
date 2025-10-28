#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/bias_correction.out"

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
VARS_ARG=${VARS:-"tas,wsiv,oht_atl,oht_pac"}
METHODS_ARG=${METHODS:-"linear_scaling,variance_scaling,quantile_mapping"}

RUN_CMD="python scripts/bias_correction.py \
    --vars \"${VARS_ARG}\" \
    --methods \"${METHODS_ARG}\" \
    --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
