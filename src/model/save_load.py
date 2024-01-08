# Save or load trained models. This required saving (or loading) both the
# trained model parameters, as well as the image_size and noise_int_to_str
# dictionary needed to instantiate the model. These are done in tandem, so that
# loading a model automatically returns the fully restored model.

from src.utils.save_load import save_file, load_file
from src.model.define_model import Net
import numpy as np
import torch

MODEL_BASEPATH = 'output/trained_models/'


def get_paired_filenames(filename):
    # If a filename is given, return the two associated files with appropriate extensions.
    if type(filename) is str:
        if len(filename) > 4 and filename[-4:] == ".pth":
            params_filename = filename
            init_filename = filename[:-4] + ".npy"
        elif len(filename) > 4 and filename[-4:] == ".npy":
            params_filename = filename[:-4] + ".pth"
            init_filename = filename
        else:
            params_filename = filename + ".pth"
            init_filename = filename + ".npy"
        return params_filename, init_filename
    else:
        return None, None


def save_model(model, filename=None, rewrite=False, basepath=MODEL_BASEPATH):
    """ Save the trained model parameters, and also the image_size and noise_int_to_str dictionary """

    def save_function_parameters(data, path):
        torch.save(data.state_dict(), path)

    def save_function_init(data, path):
        np.save(path, np.array([data.image_size, data.noise_int_to_str], dtype=object))

    # save the neural net parameters
    print('Saving the model parameters (.pth) ...')
    parameters_path = save_file(data=model, filename=filename, rewrite=rewrite,
                                basepath=basepath, extension=".pth",
                                save_function=save_function_parameters)

    # abort if that save failed
    if parameters_path is None:
        return

    # save the image_size and noise_int_to_str dictionary to the same directory
    # and filename, different extension. These are needed to initialize a new Net
    print('Saving the image_size and noise_int_to_str (.npy) ...')
    filename = parameters_path[len(MODEL_BASEPATH):-4]
    init_path = save_file(data=model, filename=filename, rewrite=rewrite,
                          basepath=basepath, extension=".npy",
                          save_function=save_function_init)

    return parameters_path, init_path


def load_model(filename=None, basepath=MODEL_BASEPATH):
    """ Load and initialize trained model """

    def load_function_params(path):
        state_dict = torch.load(path)
        return state_dict

    def load_function_init(path):
        image_size, noise_int_to_str = np.load(path, allow_pickle=True)
        return image_size, noise_int_to_str

    # If no filename is given, prompt for filenames. Otherwise, load the two associated files.
    params_filename, init_filename = get_paired_filenames(filename)
    if (params_filename, init_filename) == (None, None):
        print('Load the parameters file, with extension .pth\n')
        state_dict = load_file(
            filename=None, basepath=basepath, load_function=load_function_params)
        if state_dict is None:
            return
        print('\nLoad the initialization file, with extension .npy\n')
        model_init = load_file(
            filename=None, basepath=basepath, load_function=load_function_init)
    else:
        print('Loading the parameters file (.pth) ...')
        state_dict = load_file(
            filename=params_filename, basepath=basepath, load_function=load_function_params)
        print('\nLoading the initialization file (.npy) ...')
        model_init = load_file(
            filename=init_filename,   basepath=basepath, load_function=load_function_init)

    # Abort if either load failed.
    if (state_dict is None) or (model_init is None):
        print('Could not load both files. Aborting.')
        return

    # Build the model from the resulting data
    image_size, noise_int_to_str = model_init
    model = Net(image_size, noise_int_to_str)
    model.load_state_dict(state_dict)

    return model

# # TESTING
# save_model(my_net, filename='my_model')
# # TESTING
# new_model = load_model(filename='my_model')
# print(new_model)
