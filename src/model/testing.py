# --------- Pretend we're in the root directory ---------
import sys
import os

ROOT_DIR = '../..'
os.chdir(ROOT_DIR)
sys.path.insert(0, os.getcwd())
# -------------------------------------------------------

from src.audio.device_settings import prompt_device_selection, get_samplerate
from src.audio.make_spectrograms import generate_spectrogram
from src.audio.save_load import load_noise_sample_dict
from src.model.prepare_datasets import NoisesDataset, prepare_even_data_loaders
from src.model.define_model import Net
from src.model.train_model import train_net
from src.model.evaluate_model import accuracy_rating, plot_confusion_matrix

if __name__ == "__main__":

    ###################### TESTING MODELS ######################

    device = 0  # alternatively, use prompt_device_selection()
    samplerate = get_samplerate(device)

    # model/prepare_datasets.py

    import matplotlib.pyplot as plt

    my_recordings = load_noise_sample_dict(
        filename='100each__t-p-k-ch-tsk-click.npy')
    my_dataset = NoisesDataset(my_recordings, samplerate)
    my_train_loader, my_test_loader, my_train_dataset, my_test_dataset = prepare_even_data_loaders(
        my_dataset, samplerate, batch_size=8)

    # get some random training spectrograms from the data loader
    my_train_dataiter = iter(my_train_loader)
    my_spectrograms, my_labels = next(my_train_dataiter)
    my_batch_size = 8

    # for those training data, show spectrograms and print their labels
    fig, ax = plt.subplots(1, my_batch_size)
    for i in range(len(my_spectrograms)):
        # the 0 selects the first (only) channel
        ax[i].imshow(my_spectrograms[i][0].numpy())
    print(' '.join('{:>4s}'.format(
        my_dataset.noise_int_to_str[my_labels[j].item()]) for j in range(my_batch_size)))
    # plt.show()    # uncomment to see sample training spectrograms

    # model/define_model.py

    image_size = my_spectrograms[0].size()
    my_net = Net(image_size, my_dataset.noise_int_to_str)
    print(my_net)

    # model/train_model.py

    train_net(my_net, 20, my_train_loader, batch_progress=100)

    # model/evaluate_model.py

    accuracy_rating(my_net, my_train_loader, 'training')
    preds, targets = accuracy_rating(my_net, my_test_loader, 'test')

    plot_confusion_matrix(preds, targets, my_dataset.noise_int_to_str)
