# c4ml\_test\_runs

This repository contains a series of experiments that compare the performance of chisel4ml and hls4ml.


To reproduce the results, you will require Vitis HLS 2023.1, Python 3.10, and the mill build tool.
Then run the following commands:
- `git clone https://github.com/jurevreca12/c4ml_test_runs --recurse_submodules`
- `cd c4ml_test_runs/chisel4ml`
- `python3.10 -m venv venv`
- `source venv/bin/activate`
- `pip install -ve .[dev]`
- `make`
- `mill chisel4ml.assembly`
- `cd ..`
- `pip install -r requirements.txt`
- `source XILINX_HOME/Vitis/2023.1/settings.sh`
- `python main.py`

To run any particular experiments, use: `python main.py -name linear_layer_var_in_features_exp`.
