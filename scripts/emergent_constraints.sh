#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/icefreearcticml"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/emergent_constraints.out"

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
PREDICTOR_VAR_ARG="tas"
TARGET_VAR_ARG="ssie"
HIST_START_ARG="1980"
HIST_END_ARG="2014"
FUT_START_ARG="2030"
FUT_END_ARG="2060"
METHOD_ARG="linear"          # linear | mlp
TARGET_STAT_ARG="mean"       # mean | trend
MODEL_NAME_ARG="all"
TIME_VARYING_FLAG=""

# TIME_VARYING can be set to 1/true/TRUE to enable the flag
if [[ "${TIME_VARYING}" == "1" || "${TIME_VARYING}" == "true" || "${TIME_VARYING}" == "TRUE" ]]; then
  TIME_VARYING_FLAG="--time-varying"
fi

RUN_CMD="python scripts/emergent_constraints.py \
  --predictor-var ${PREDICTOR_VAR_ARG} \
  --target-var ${TARGET_VAR_ARG} \
  --hist-start ${HIST_START_ARG} \
  --hist-end ${HIST_END_ARG} \
  --fut-start ${FUT_START_ARG} \
  --fut-end ${FUT_END_ARG} \
  ${TIME_VARYING_FLAG} \
  --method \"${METHOD_ARG}\" \
  --target-stat ${TARGET_STAT_ARG} \
  --model-name ${MODEL_NAME_ARG} \
  --save-dir ${OUTPUT_DIR}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
