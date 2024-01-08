# Train the convolutional neural network.

# Some of the model construction, training, and evaluation code here has
# been adapted from a [pytorch
# tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py).

import torch.nn as nn
import torch.optim as optim

def train_net(net, epochs, train_loader, batch_progress=50):
    """ Use training data from train_loader to train net for a number of epochs,
    using a cross entropy loss function and Adam as the optimizer. """
    
    # the loss function and optimizing method
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    batch_num = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        batch_running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accrue loss for printing
            batch_running_loss += loss.item()
            
            # print progress every batch_progress batches
            if batch_num % batch_progress == batch_progress-1:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(
                  epoch + 1, i + 1, batch_running_loss / batch_progress))
                batch_running_loss = 0.0
                batch_num = 0
            
            batch_num += 1
        
    print('Finished Training')

# # TESTING
# my_net = Net(my_spectrogram.size(), my_dataset.noise_int_to_str)
# train_net(my_net, 50, my_train_loader, batch_progress=100)