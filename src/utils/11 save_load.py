## Saving/loading recordings and models
# Functions to save (and load) recordings and models, if desired.
import os
# Some helper functions to save or load generic files:
def save_file(data, filename, rewrite, basepath, extension, save_function): 
    """ Save data to basepath/filename.extension using save_function """
    
    # Get a nonempty filename. Require it to be unique unless rewrite=True.
    while True:
        if filename is None:
            print('Enter filename to write to (leave blank to cancel):')
            filename = input()

        if filename == '':
            print("No filename provided. Aborting.")
            return

        # Add the extension if it isn't already included
        if filename[-4:] != extension:
            filename += extension

        # Make the base directory if it doesn't already exist
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        # Don't accidentally overwrite an existing file
        valid_filename = True
        if not rewrite:
            with os.scandir(basepath) as entries:
                for entry in entries:
                    if entry.name == filename:
                        print('File {} already exists. Use rewrite=True to overwrite.\n'.format(
                        './' + basepath + filename))
                        filename = None
                        valid_filename = False
        
        if valid_filename:
            break

    # Write the file, if we've made it this far
    path = basepath + filename
    save_function(data, path)
    print('Data saved to', path)
    
    return path
    
def load_file(filename, basepath, load_function):
    """ Load a file from basepath/filename """
    
    # check if the base directory exists
    if not os.path.exists(basepath):
        print("Base directory {} doesn't exist. Aborting.".format(basepath))
        return 

    # Get the filename if one wasn't provided, abort if empty
    if filename is None:
        # display the available files for user convenience
        print('Files in {} include:'.format(basepath))
        with os.scandir(basepath) as entries:
            for entry in entries:
                print(entry.name)

        print('\nEnter filename to load:')
        filename = input()
        if filename == '':
            print('Aborting.')
            return
        
    # Abort if the file doesn't exist
    with os.scandir(basepath) as entries:
        if filename not in [ entry.name for entry in entries ]:
            print('File does not exist. Aborting.')
            return 
   
    # Load the file, if we've made it this far.
    path = basepath + filename
    loaded_data = load_function(path)
    print('File loaded:', path)

    return loaded_data
# Save or load noise sample dictionaries, with audio data:
DICT_BASEPATH = 'saved_noise_sample_dictionaries/'

def save_noise_sample_dict(noise_data_dict, filename=None, rewrite=False, basepath=DICT_BASEPATH): 
    """ Save the dictionary of noise sample recordings """
    
    def save_function(data, path):
        np.save(path, data)
        
    return save_file(data=noise_data_dict, filename=filename, rewrite=rewrite,
                    basepath=basepath, extension=".npy", save_function=save_function)
    
def load_noise_sample_dict(filename=None, basepath=DICT_BASEPATH):
    """ Load a dictionary of noise sample recordings """
    
    def load_function(path):
        return np.load(path, allow_pickle=True).item()
        
    return load_file(filename=filename, basepath=basepath, load_function=load_function)
# # TESTING
# save_noise_sample_dict(my_recordings)
# my_loaded_file = load_noise_sample_dict(filename='my_recordings.npy')
# my_loaded_file
# Save or load trained models. This required saving (or loading) both the trained model parameters, as well as the image_size and noise_int_to_str dictionary needed to instantiate the model. These are done in tandem, so that loading a model automatically returns the fully restored model.
MODEL_BASEPATH = 'trained_models/'

def get_paired_filenames(filename):
    # If a filename is given, return the two associated files with appropriate extensions.
    if type(filename) is str:
        if   len(filename) > 4 and filename[-4:] == ".pth":
            params_filename = filename
            init_filename   = filename[:-4] + ".npy"
        elif len(filename) > 4 and filename[-4:] == ".npy":
            params_filename = filename[:-4] + ".pth"
            init_filename   = filename
        else:
            params_filename = filename + ".pth"
            init_filename   = filename + ".npy"
        return params_filename, init_filename
    else:
        return None, None

def save_model(model, filename=None, rewrite=False, basepath=MODEL_BASEPATH): 
    """ Save the trained model parameters, and also the image_size and noise_int_to_str dictionary """
    
    def save_function_parameters(data, path):
        torch.save(data.state_dict(), path)
        
    def save_function_init(data, path):
        np.save(path, (data.image_size, data.noise_int_to_str))
    
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
        state_dict = load_file(filename=None, basepath=basepath, load_function=load_function_params)
        if state_dict is None:
            return
        print('\nLoad the initialization file, with extension .npy\n')
        model_init = load_file(filename=None, basepath=basepath, load_function=load_function_init)
    else:
        print('Loading the parameters file (.pth) ...')
        state_dict = load_file(filename=params_filename, basepath=basepath, load_function=load_function_params)
        print('\nLoading the initialization file (.npy) ...')
        model_init = load_file(filename=init_filename,   basepath=basepath, load_function=load_function_init)
    
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