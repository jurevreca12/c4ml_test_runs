import os
from models.train import train_float_lhc_model
from models.train import train_float_mnist_model

if __name__ == "__main__":
    #trained_model_lhc, test_data_lhc, final_acc_lhc = train_float_lhc_model()
    #if not os.path.exists("plots/lhc_model_float"):
    #    os.makedirs("plots/lhc_model_float")
    #with open("plots/lhc_model_float/acc.log", 'w') as f:
    #    f.write(str(final_acc_lhc))

    trained_model_cnn, test_data_cnn, final_acc_cnn = train_float_mnist_model()
    if not os.path.exists("plots/cnn_mnist_model_float"):
        os.makedirs("plots/cnn_mnist_model_float")
    with open("plots/cnn_mnist_model_float/acc.log", 'w') as f:
        f.write(str(final_acc_cnn))
