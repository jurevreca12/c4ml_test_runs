#!/bin/bash
source ${XILINX_DIR}/Vitis/${XILINX_VERSION}/settings64.sh
ulimit -s 262144
if [[ "$*" == *"--gen-graphs"* ]]; then
    python /workspace/gen_graphs.py
else
    echo "python /workspace/main.py $@"
    python -u /workspace/main.py "$@"
fi

