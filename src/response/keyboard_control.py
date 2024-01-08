# Keyboard control: up down left right escape
# We want to play a videogame, like Crypt of the Necrodancer, with this noise control. Let's introduce some basic keyboard control.

from .print_noise import print_noise
import pyautogui


########## NOISE-TO-KEYBOARD MAPPING ##########

KEYBOARD_MAPPING = {
    't': 'up',
    'k': 'right',
    'p': 'down',
    'tsk': 'left',
    'cluck': 'escape',
}

# basic keyboard control


def press_key(noise_heard):
    try:
        pyautogui.press(KEYBOARD_MAPPING[noise_heard])
    except:
        print('No keyboard mapping for "{}"'.format(noise_heard))
    print_noise(noise_heard)


# For best performance, define a model that only recognizes these noises, even
# if we recorded more earlier. E.g.,
# my_recordings_subset = {
#     k: v for k, v in my_recordings.items() if k in KEYBOARD_MAPPING.keys()}
# necrodancer_model, _ = build_beatbot(
#     device=device, skip_recording=True, starting_noise_data=my_recordings_subset)
