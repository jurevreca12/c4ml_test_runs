python main.py -n 1 -exp 0 >python0.log 2>&1 &   # linear_layer_var_in_features
python main.py -n 1 -exp 1 >python1.log 2>&1 &   # linear_layer_var_out_featuers
python main.py -n 1 -exp 2 >python2.log 2>&1 &   # linear_layer_var_iq
python main.py -n 1 -exp 3 >python3.log 2>&1 &   # linear_layer_var_wq

#python main.py -n 3 -exp 4 >python4.log 2>&1 &   # conv_layer_var_input_ch
python main.py -n 2 -exp 5 >python5.log 2>&1 &   # conv_layer_var_output_ch
python main.py -n 2 -exp 6 >python6.log 2>&1 &   # dw_conv_layer_var_output_ch
python main.py -n 2 -exp 7 >python7.log 2>&1 &   # conv_layer_var_iq
python main.py -n 2 -exp 8 >python8.log 2>&1 &   # conv_layer_var_wq

#python main.py -n 3 -exp 9 >python9.log 2>&1 &   # maxpool_layer_var_input_size
python main.py -n 1 -exp 10 >python10.log 2>&1 &  # maxpool_layer_var_channels
python main.py -n 1 -exp 11 >python11.log 2>&1 &  # maxpool_layer_var_kernel_size
