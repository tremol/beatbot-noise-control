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
        if filename not in [entry.name for entry in entries]:
            print('File does not exist. Aborting.')
            return

    # Load the file, if we've made it this far.
    path = basepath + filename
    loaded_data = load_function(path)
    print('File loaded:', path)

    return loaded_data