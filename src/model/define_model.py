# Define the convolutional neural network that we will train and use for noise recognition.

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, image_size, noise_int_to_str):
        super(Net, self).__init__()
        
        # the spectrogram image size is needed to compute layer sizes
        self.image_size = image_size
        
        # the dictionary of noise labels is needed for translating predictions in the final layer
        self.noise_int_to_str = noise_int_to_str
        
        
        # image_size is a 2-tuple, the expected dimensions of each spectrogram
        # .... or a 3-tuple, if the channel has already been added
        if   len(image_size) == 2:
            h, w = image_size
        elif len(image_size) == 3:
            channel, h, w = image_size
        
        # number of output nodes, (square) kernel size, and pool size per convolution layer,
        # assuming the stride for pooling is the same as the pool size
        kernels = [3, 3]
        pool = 2
        
        # compute the number of input nodes for the first dense layer
        h_out, w_out = h, w
        for k in kernels:
            # the convolution.
            h_out += -k + 1
            w_out += -k + 1
            
            # the pool. (from help(torch.nn.MaxPool2d))
            h_out = int( (h_out - pool) / pool + 1 )
            w_out = int( (w_out - pool) / pool + 1 )
            
        self.image_out = h_out * w_out
        
        # define the layers. The numbers of nodes chosen do not have deep thought behind them.
        self.conv0 = nn.Conv2d(1, 32, kernels[0])
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(32, 10, kernels[1])
        self.fc0 = nn.Linear(10 * self.image_out, 50)
        self.fc1 = nn.Linear(50, 10)
        # number of output nodes for final dense layer: the number of noise types        
        self.fc2 = nn.Linear(10, len(noise_int_to_str))
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 10 * self.image_out)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# # TESTING
# my_net = Net(my_spectrogram.size(), my_dataset.noise_int_to_str)