# Recording audio data for training/testing


from src.audio.listen import listen, listen_and_process
from src.utils.input import get_nonneg_int_input
import sys
import time


########## A listener to record training/testing data. ##########


def print_progress(noise_data_dict):
    """ A helper to print recording progress. """
    print('Noises recorded so far:', {k: len(v)
                                      for k, v in noise_data_dict.items()})


# Use the generic listening function and add a user interface to record audio data for model training
def record_model_data(device, starting_noise_data={}):
    """ Prompts the user to label and record noise samples. Returns a dictionary with labels as keys
    and lists of flattened numpy arrays (one array per noise sample) as values. """

    noise_data_dict = starting_noise_data.copy()
    noise_count = 0

    # a helper to gather audio samples and increment the progress counter
    def gather_and_progress(label, total):
        def _gather_and_progress(rec):
            nonlocal noise_count

            listen.all_noises.append(rec)
            noise_count += 1
            sys.stdout.write(str(noise_count) + ', ')
            sys.stdout.flush()
            # print(noise_count, end=', ')

        return _gather_and_progress

    # Ask user how many samples to record
    same_num_for_each = False
    print(
        'Would you like to record the same number of samples for each noise type? [Y/n]')
    answer = input()
    if answer == '' or 'y' in answer.lower():
        same_num_for_each = True
        print('How many noise samples would you like to record for each?')
        num = get_nonneg_int_input()
    else:
        print('We recommend recording similar numbers for each type, to avoid biasing the model.\n')

    # ask user for noise labels, and listen and record the desired number of samples
    while True:
        print_progress(noise_data_dict)
        print('Enter text label for next noise (leave blank to exit):')
        label = input()
        if not label:
            return noise_data_dict
        if label in noise_data_dict:
            print('You have already recorded {} samples of this noise. You may now record more.'.format(
                len(noise_data_dict[label]))
            )
        if not same_num_for_each:
            print('How many noise samples would you like to record?')
            num = get_nonneg_int_input()
            if num == 0:
                continue

        # clear_output()  # clear jupyter output
        print_progress(noise_data_dict)
        time.sleep(0.5)  # avoid catching noise from keyboard
        print('Please start recording.\n')
        print('"{}" noises recorded (out of {}): '.format(label, num))
        noise_count = 0
        listen_and_process(processing_function=gather_and_progress(label, num),
                           device=device,
                           stop_condition=lambda: noise_count >= num,
                           print_after_processing=None)
        print('')

        # save the list of recorded noises
        if label in noise_data_dict:
            noise_data_dict[label] += listen.all_noises.copy()
        else:
            noise_data_dict[label] = listen.all_noises.copy()

    return noise_data_dict


# TESTING

# my_recordings = record_model_data(device=0)
# print(my_recordings)
