#!/bin/bash
source ${XILINX_DIR}/Vitis/${XILINX_VERSION}/settings64.sh
ulimit -s 262144
echo "python /workspace/main.py $*"
python /workspace/main.py "$@"
