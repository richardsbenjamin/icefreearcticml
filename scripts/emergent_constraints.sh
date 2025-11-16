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

############################################
# Environment overrides with defaults (TVEC)
FUT_VAR_ARG="ssie"
HIST_VAR_ARG="ssie,tas,wsiv,oht_atl,oht_pac" #"ssie,tas,wsiv"
OBS_START_ARG="1979" 
OBS_END_ARG="2018" # 2023 for ssie, tas, wsiv as HIST_VAR_ARG; 2018 for oht_atl, oht_pac
CALIB_START_ARG="2024"
CALIB_END_ARG="2099"
WINDOW_ARG="5"
MODEL_TYPE_ARG="linear"

RUN_CMD="python scripts/emergent_constraints.py \
  --hist-var ${HIST_VAR_ARG} \
  --fut-var ${FUT_VAR_ARG} \
  --obs-start ${OBS_START_ARG} \
  --obs-end ${OBS_END_ARG} \
  --calib-start ${CALIB_START_ARG} \
  --calib-end ${CALIB_END_ARG} \
  --window ${WINDOW_ARG} \
  --model-type ${MODEL_TYPE_ARG} \
  --save-dir ${OUTPUT_DIR}"


# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
else
    ${RUN_CMD}
fi
