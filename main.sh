#!/bin/bash
source ${XILINX_DIR}/Vitis/${XILINX_VERSION}/settings64.sh
ulimit -s 262144
if [[ "$*" == *"--gen-graphs"* ]]; then
    shift
    echo "python /workspace/gen_graphs.py $@"
    python /workspace/gen_graphs.py "$@"
elif [[ "$*" == *"--gen-acc-graphs"* ]]; then
    shift
    echo "python /workspace/gen_acc_graphs.py $@"
    python /workspace/gen_acc_graphs.py "$@"
elif [[ "$*" == *"--train-float"* ]]; then
    echo "python /workspace/train_float.py"
    python /workspace/train_float.py
else
    echo "python /workspace/main.py $@"
    python -u /workspace/main.py "$@"
fi

