import torch

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:   
                print(f'[{epoch + 1}/{epochs}, {i + 1:3d}/{len(train_loader)}] - loss: {running_loss / 2000:.3f}')
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
