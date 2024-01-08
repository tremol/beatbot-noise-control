# Functions to save (and load) recordings, if desired.

from src.utils.save_load import save_file, load_file
import numpy as np

# Save or load noise sample dictionaries, with audio data:
DICT_BASEPATH = 'output/saved_noise_samples/'


def save_noise_samples(noise_data_dict, filename=None, rewrite=False, basepath=DICT_BASEPATH):
    """ Save the dictionary of noise sample recordings """

    def save_function(data, path):
        np.save(path, data)

    return save_file(data=noise_data_dict, filename=filename, rewrite=rewrite,
                     basepath=basepath, extension=".npy", save_function=save_function)


def load_noise_samples(filename=None, basepath=DICT_BASEPATH):
    """ Load a dictionary of noise sample recordings """

    def load_function(path):
        return np.load(path, allow_pickle=True).item()

    return load_file(filename=filename, basepath=basepath, load_function=load_function)

# # TESTING
# save_noise_samples(my_recordings)
# my_loaded_file = load_noise_samples(filename='my_recordings.npy')
# my_loaded_file
