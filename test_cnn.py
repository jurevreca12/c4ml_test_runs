import os
from cnn_model import get_cnn_model
from mnist_data import get_data_loaders
from train import train_model, eval_model, merge_batchnorm
from chisel4ml import transform
from test_model import test_chisel4ml, test_hls4ml
import torch
import onnx
import multiprocessing
from server import create_server
from multiprocessing.pool import ThreadPool
import argparse
import json
from parse_reports import parse_reports

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def train_quantized_mnist_model(bitwidth, work_dir):
    model = get_cnn_model(bitwidth = bitwidth, use_bn=True)
    model_nobn = get_cnn_model(bitwidth = bitwidth, use_bn=False)
    train_loader, test_loader = get_data_loaders(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        epochs=5,
        device=device
    )
    print(f"ACCURACY WITH BN {bitwidth}:")
    eval_model(model, test_loader, device)

    print("MERGING BATCHNORM TO ACTIVE LAYERS")
    model.eval()
    model_nobn.load_state_dict({k: v for k, v in model.state_dict().items() if 'bn' not in k})
    for layer in (model.conv0, model.conv1):
        merge_batchnorm(layer.conv, layer.bn)  
    for layer in (model.dense0, model.dense1):
        merge_batchnorm(layer.dense, layer.bn)  
    model_nobn.load_state_dict({k: v for k, v in model.state_dict().items() if 'bn' not in k}, strict=False)
    print('BN layers fused.')
    print("RETRAINING FUSED MODEL")
    train_model(
        model=model_nobn,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model_nobn.parameters(), lr=0.001),
        epochs=5,
        device=device
    )
    print(f"FINAL ACCURACY {bitwidth} (NO BN):")
    final_acc = eval_model(model_nobn, test_loader, device)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    with open(f"{work_dir}/acc.log", 'w') as f:
        f.write(f"Final accuracy-{bitwidth}:\n")
        f.write(f"{str(final_acc)}\n")
    # return trained model and one batch of data for testing
    torch_tensor = next(iter(test_loader))[0]
    return model_nobn, torch_tensor.detach().numpy()

lock = multiprocessing.Lock()
def test_cnn(bitwidth):
    global lock
    work_dir = f'circuits/mnist/cnn{bitwidth}/'
    brevitas_model, test_data = train_quantized_mnist_model(bitwidth, work_dir)
    lock.acquire()
    qonnx_model = transform.brevitas_to_qonnx(brevitas_model, brevitas_model.ishape)
    lock.release()
    os.makedirs(f"{work_dir}/qonnx")
    onnx.save(qonnx_model.model, f"{work_dir}/qonnx/model.onnx")
    test_chisel4ml(qonnx_model, brevitas_model, test_data, f"{work_dir}/c4ml", base_dir=SCRIPT_DIR, top_name="ProcessingPipeline")
    c4ml_res = parse_reports(f"{work_dir}/c4ml/")
    test_hls4ml(qonnx_model, work_dir=f"{work_dir}/hls4ml", base_dir=SCRIPT_DIR)
    hls4ml_res = parse_reports(f"{work_dir}/hls4ml/")
    return {'work_dir': work_dir, 'chisel4ml': c4ml_res, 'hls4ml': hls4ml_res}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cnn_train")
    parser.add_argument(
        '--bitwidth',
        '-bw',
        type=int,
        default=2,
        help='The bitwidth of the parameters of the model.'
    )
    args = parser.parse_args()
    print(f"Training CNN model with bitwdith: {args.bitwidth}")
    create_server('chisel4ml/out/chisel4ml/assembly.dest/out.jar', 1)
    result = test_cnn(args.bitwidth)
    ser_res = json.dumps(results)
    with open(f'{SCRIPT_DIR}/results_mnist_bw{args.bitwidth}.json', 'w') as f:
        f.write(ser_res)
