# the most basic response function

import sys

def print_noise(noise_heard):
    """ Print the string label of the noise that has been heard. """
    # print(noise_heard, end=", ")
    sys.stdout.write(noise_heard + ', ')
    sys.stdout.flush()