# Functions to prompt and set input device and related parameters (e.g., sample rate)

import sounddevice as sd


def get_samplerate(device):
    samplerate = sd.query_devices(device, 'input')['default_samplerate']
    return samplerate

def is_input_device(device):
    try:
        sd.check_input_settings(device)
        return True
    except sd.PortAudioError:
        print('Selected device is not an input device.')
        return False

def prompt_device_selection():
    """ Prompt user to select input device. Default: 0. """

    available_devices = sd.query_devices()

    while True:
        print('Your audio devices are:')
        print(available_devices)
        print('Enter the number of your preferred input device [0]:')
        try:
            device = int(input() or "0")
            if is_input_device(device):
                break
        except ValueError:
            print('Not a valid device. Enter only the device number (e.g., 0).')

    return device, get_samplerate(device)


  # this is enough to make it all work — just
