#python main.py -n 1 -name linear_layer_var_in_features_exp >logs/python0.log 2>&1 &
#python main.py -n 1 -name linear_layer_var_out_features_exp >logs/python1.log 2>&1 &
#python main.py -n 1 -name linear_layer_var_iq_exp >logs/python2.log 2>&1 &
#python main.py -n 1 -name linear_layer_var_wq_exp >logs/python3.log 2>&1 &

python main.py -n 2 -name conv_layer_var_input_ch_exp >logs/python4.log 2>&1 
#python main.py -n 2 -name conv_layer_var_output_ch_exp >logs/python5.log 2>&1 &
#python main.py -n 1 -name conv_layer_var_iq_exp >logs/python6.log 2>&1 &
#python main.py -n 1 -name conv_layer_var_wq_exp >logs/python7.log 2>&1 &

#python main.py -n 1 -name maxpool_layer_var_input_size_exp >logs/python8.log 2>&1 &
#python main.py -n 1 -name maxpool_layer_var_channels_exp  >logs/python9.log 2>&1 &
#python main.py -n 1 -name maxpool_layer_var_kernel_size_exp >logs/python10.log 2>&1 & 
#python main.py -n 1 -name maxpool_layer_var_iq_exp >logs/python11.log 2>&1 & 

python main.py -n 2 -name cnn_mnist_model_var_bitwidth_exp >logs/python12.log 2>&1 
python main.py -n 2 -name cnn_mnist_model_var_prune_rate_exp >logs/python13.log 2>&1 
