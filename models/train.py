import torch
from torch.nn.utils import prune
import brevitas.nn as qnn
from models.cnn_model import get_cnn_model
from models.mnist_data import get_data_loaders


def print_sparsity_by_layer(model):
	print(
		"Sparsity in conv0.conv.weight: {:.2f}%".format(
			100. * float(torch.sum(model.conv0.conv.weight == 0))
			/ float(model.conv0.conv.weight.nelement())
		)
	)
	print(
		"Sparsity in conv1.conv.weight: {:.2f}%".format(
			100. * float(torch.sum(model.conv1.conv.weight == 0))
			/ float(model.conv1.conv.weight.nelement())
		)
	)
	print(
		"Sparsity in dense0.dense.weight: {:.2f}%".format(
			100. * float(torch.sum(model.dense0.dense.weight == 0))
			/ float(model.dense0.dense.weight.nelement())
		)
	)
	print(
		"Sparsity in dense1.dense.weight: {:.2f}%".format(
			100. * float(torch.sum(model.dense1.dense.weight == 0))
			/ float(model.dense1.dense.weight.nelement())
		)
	)
	print(
		"Global sparsity: {:.2f}%".format(
			100. * float(
				torch.sum(model.conv0.conv.weight == 0)
				+ torch.sum(model.conv1.conv.weight == 0)
				+ torch.sum(model.dense0.dense.weight == 0)
				+ torch.sum(model.dense1.dense.weight == 0)
			)
			/ float(
				model.conv0.conv.weight.nelement()
				+ model.conv1.conv.weight.nelement()
				+ model.dense0.dense.weight.nelement()
				+ model.dense1.dense.weight.nelement()
			)
		)
	)


def prune_model_global_unstructured(model, prune_rate, print_sparsity=False):
    parameters_to_prune = (
        (model.conv0.conv, 'weight'),
        (model.conv1.conv, 'weight'),
        (model.dense0.dense, 'weight'),
        (model.dense1.dense, 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune, 
        pruning_method=prune.L1Unstructured,
        amount=prune_rate
    )

    # Remove prunning re-parameterization
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')


def train_model(model, train_loader, criterion, optimizer, epochs, device, prune_rate):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            inputs = inputs * 255
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            
            prune_model_global_unstructured(model, prune_rate)

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}/{epochs}, {i + 1:3d}/{len(train_loader)}] - loss: {running_loss / 2000:.5f}')
                running_loss = 0.0
    print('Finished Training')


def eval_model(model, test_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = float(correct) / total
    print(f'Accuracy of the network on the 10000 test images: {100 * accuracy} %')
    return accuracy


def merge_batchnorm(act, bn, ltype='conv'):
    if ltype == "conv":
        w_act = act.weight.clone().view(act.out_channels, - 1)
    else:
        w_act = act.weight.clone()
    inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w_bn = torch.diag(inv)
    new_w = torch.mm(w_bn, w_act).view(act.weight.size())
    if act.bias is not None:
        b_act = act.bias
    else:
        b_act = torch.zeros(act.weight.size(0))
    new_b = ((b_act - bn.running_mean) * inv) + bn.bias
    act.weight = torch.nn.Parameter(new_w)
    act.bias = torch.nn.Parameter(new_b)


def train_quantized_mnist_model(bitwidth, prune_rate=0.0):
    model = get_cnn_model(bitwidth=bitwidth, use_bn=True)
    model_nobn = get_cnn_model(bitwidth=bitwidth, use_bn=False)
    train_loader, test_loader = get_data_loaders(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        epochs=5,
        device=device,
        prune_rate=prune_rate
    )
    print_sparsity_by_layer(model)

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
        device=device,
        prune_rate=prune_rate
    )
    print_sparsity_by_layer(model)
    print(f"FINAL ACCURACY {bitwidth} (NO BN):")
    final_acc = eval_model(model_nobn, test_loader, device)
    # return trained model and one batch of data for testing
    torch_tensor = next(iter(test_loader))[0]
    return model_nobn, torch_tensor.detach().numpy(), final_acc
