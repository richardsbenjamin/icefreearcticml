#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation.out"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

# Environment overrides with defaults
JOBS_ARG=""                     # REQUIRED: comma-separated .joblib paths
ICE_DS_ARG="${OUTPUT_DIR}/ice_free_year_ds.nc"
LIANG_START_ARG="1980-01-01"
LIANG_END_ARG="2060-01-01"
X_LIANG_ARG="wsiv,tas,oht_atl,oht_pac"
Y_LIANG_ARG="ssie"

# Guard: require JOBS
if [[ -z "${JOBS_ARG}" ]]; then
  echo "ERROR: Set JOBS to a comma-separated list of experiment .joblib files" >&2
  exit 1
fi

RUN_CMD="python scripts/evaluation.py \
  --jobs ${JOBS_ARG} \
  --ice-ds ${ICE_DS_ARG} \
  --liang-start ${LIANG_START_ARG} \
  --liang-end ${LIANG_END_ARG} \
  --x-liang ${X_LIANG_ARG} \
  --y-liang ${Y_LIANG_ARG} \
  --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
