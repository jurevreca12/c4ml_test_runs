from cnn_model import get_cnn_model
import onnx
from mnist_data import get_data_loaders
from train import train_model, eval_model
from chisel4ml import transform

import torch
from  brevitas.nn.utils import merge_bn

model = get_cnn_model(bitwidth = 4, use_bn=True)
model_nobn = get_cnn_model(bitwidth = 4, use_bn=False)
train_loader, test_loader = get_data_loaders(batch_size=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(
    model=model,
    train_loader=train_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    epochs=1,
    device=device
)
eval_model(model, test_loader, device)

model.eval()
model_nobn.load_state_dict({k: v for k, v in model.state_dict().items() if 'bn' not in k})

def merge_batchnorm(act, bn, ltype='conv'):
	if ltype == "conv":
		w_act = act.weight.clone().view(act.out_channels, -1)
	else:
		w_act = act.weight.clone()
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	new_w = torch.mm(w_bn, w_act).view(act.weight.size())
	if act.bias is not None:
		b_act = act.bias
	else:
		b_act = torch.zeros(act.weight.size(0))
	new_b = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	act.weight = torch.nn.Parameter(new_w)
	act.bias = torch.nn.Parameter(new_b)

for layer in (model.conv0, model.conv1):
    merge_batchnorm(layer.conv, layer.bn)  
for layer in (model.dense0, model.dense1):
    merge_batchnorm(layer.dense, layer.bn)  
model_nobn.load_state_dict({k: v for k, v in model.state_dict().items() if 'bn' not in k}, strict=False)
print('layer fused ')
eval_model(model_nobn, test_loader, device)
train_model(
    model=model_nobn,
    train_loader=train_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model_nobn.parameters(), lr=0.001),
    epochs=1,
    device=device
)
eval_model(model_nobn, test_loader, device)

qonnx_model = transform.brevitas_to_qonnx(model, model.ishape)
onnx.save(qonnx_model.model, 'test_cnn_model.onnx')

import pdb; pdb.set_trace()

