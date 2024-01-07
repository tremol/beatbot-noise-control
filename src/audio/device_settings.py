# Functions to prompt and set input device and related parameters (e.g., sample rate)

import sounddevice as sd


def get_samplerate(my_device):
    samplerate = sd.query_devices(my_device, 'input')['default_samplerate']
    return samplerate


def prompt_device_selection():
    """ Prompt user to select input device. Default: 0. """

    available_devices = sd.query_devices()
    valid_numbers = list(range(len(available_devices)))

    while True:
        print('Your audio devices are:')
        print(available_devices)
        print('Enter the number of your preferred input device [0]:')
        try:
            device = input()
            if device == '':
                device = 0

            device = int(device)
            if device in valid_numbers:
                break
            else:
                raise ValueError
        # triggers if invalid number entered, or if not a number entered
        except ValueError:
            print('Not a valid device. Enter only the device number (e.g., 0).')

    return device, get_samplerate(device)


  # this is enough to make it all work — just
