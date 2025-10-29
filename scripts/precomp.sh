#!/bin/bash

# Local paths
HOME_DIR="/content"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"

# Input paths
DATA_DIR="${MODULE_DIR}/data/"

# Output paths
OUTPUT_FILE="${OUTPUT_DIR}/precomp.out" # Path to catch errors and logs


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
VARS_ARG="tas,wsiv,oht_atl,oht_pac"
BIAS_START_ARG="1980-01-01"
BIAS_END_ARG="2014-01-01"

RUN_CMD="python scripts/precomp.py \
    --vars ${VARS_ARG} \
    --bias-start ${BIAS_START_ARG} \
    --bias-end ${BIAS_END_ARG} \
    --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi