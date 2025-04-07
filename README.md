# c4ml\_test\_runs

First start docker with:
	```docker compose run --remove-orphans --build c4ml_test_runs```
	```docker compose run -e XILINX_DIR=... XILINX_VERSION=... --remove-orphans --build c4ml_test_runs```
    ```docker compose run --build c4ml_test_runs -n 1 -name cnn_mnist_model_var_bitwidth_exp -d```

Next, to run a single experiment (e.g linear_layer_var_iq_exp):
	```python main.py -n 1 -name linear_layer_var_iq_exp```

To run all experiments:
	```python main.py```


