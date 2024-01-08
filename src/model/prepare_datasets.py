# Neural net: preparing datasets From a dictionary of labeled noise sample
# recordings, prepare the datasets and data loaders needed to train and test the
# model. Some of the model construction, training, and evaluation code here has
# been adapted from a [pytorch
# tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py).
# Data loaders partly adapted from
# https://stackoverflow.com/questions/53916594/typeerror-object-of-type-numpy-int64-has-no-len

from src.audio.make_spectrograms import N_MELS, generate_spectrogram
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import numpy as np

# Prepare the noise data for handing to the convolutional neural network, for training and testing


class NoisesDataset(Dataset):
    """ Noises dataset. Takes a dictionary of recordings and returns spectrograms when data is requested. 
    A channel dimension is added to each spectrogram, as needed for the CNN. """

    def __init__(self, noise_data_dict, samplerate, n_mels=N_MELS):
        """ Initialization: 
        Takes a dictionary of noise samples, with labels as keys and lists of
        flattened numpy arrays (one array per noise sample) as values. 
        Computes spectrograms for each. """

        self.noise_data_dict = noise_data_dict
        self.noise_samples = []
        self.labels = []

        self.noise_str_to_int = {}  # correspondences between integer and string labels
        self.noise_int_to_str = {}
        i = 0

        for label, list_of_arrays in noise_data_dict.items():
            # extract samples and labels from the dictionary
            num_samples = len(noise_data_dict[label])
            self.noise_samples += noise_data_dict[label]
            self.labels += [label] * num_samples

            # assign a unique integer to each string label
            if label not in self.noise_str_to_int:
                self.noise_str_to_int[label] = i
                self.noise_int_to_str[i] = label
                i += 1

        # compute spectrograms, and convert labels to integers
        self.spectrograms = [generate_spectrogram(s, samplerate, n_mels)
                             for s in self.noise_samples]
        self.labels = [self.noise_str_to_int[L] for L in self.labels]

    def __len__(self):
        " Return the total number of samples "
        return len(self.labels)

    def __getitem__(self, sample_index):
        " Return one sample of data "
        # Load data and get (integer) label
        # Note the CNN will expect the first tensor dimension to be the channel, hence the unsqueeze
        X = self.spectrograms[sample_index].unsqueeze(0)
        y = self.labels[sample_index]

        return X, y


def prepare_even_data_loaders(full_dataset, samplerate, training_fraction=0.8, batch_size=8):
    """ Prepare data loaders for training and testing of the model, including 
    training_fraction of each type in the training dataset. """

    train_dataset = NoisesDataset({}, samplerate)
    test_dataset = NoisesDataset({}, samplerate)

    # iterate through noise labels, adding training_fraction of each
    # to the training dataset/loader

    unique_int_labels = list(set(full_dataset.labels))
    dataset_element_labels = np.array([d[1] for d in full_dataset])

    for i in unique_int_labels:
        # get an array of indices, of noise samples with label i
        i_indices = np.nonzero(dataset_element_labels == i)[0]
        num_samples = len(i_indices)
        train_size = int(training_fraction * num_samples)
        test_size = num_samples - train_size

        # make datasets for just that noise label
        i_dataset = Subset(full_dataset, i_indices)
        i_train_dataset, i_test_dataset = random_split(
            i_dataset, [train_size, test_size])

        # add these datasets to the overall collections
        train_dataset = ConcatDataset([train_dataset, i_train_dataset])
        test_dataset = ConcatDataset([test_dataset,  i_test_dataset])

    # create the data loaders
    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 1,
    }
    train_loader = DataLoader(dataset=train_dataset, **train_params)
    test_loader = DataLoader(dataset=test_dataset)

    return train_loader, test_loader, train_dataset, test_dataset


# # TESTING prepare_even_data_loaders
# my_dataset = NoisesDataset(my_recordings, samplerate)
# my_train_loader, my_test_loader, my_train_dataset, my_test_dataset = prepare_even_data_loaders(my_dataset, samplerate, batch_size=8)

# # get some random training spectrograms
# my_train_dataiter = iter(my_train_loader)
# my_spectrograms, my_labels = next(my_train_dataiter)
# my_batch_size = 8

# # show spectrograms and print labels
# fig, ax = plt.subplots(1, my_batch_size)
# for i in range(len(my_spectrograms)):
#     ax[i].imshow(my_spectrograms[i][0].numpy()) # the 0 selects the first (only) channel
# print(' '.join('{:>4s}'.format(my_dataset.noise_int_to_str[my_labels[j].item()]) for j in range(my_batch_size)))
