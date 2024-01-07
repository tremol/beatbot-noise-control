# Keyboard control: up down left right escape
# We want to play a videogame, like Crypt of the Necrodancer, with this noise control. Let's introduce some basic keyboard control.
import pyautogui
# basic keyboard control


def press_key(noise_heard):
    keyboard_mapping = {
        't': 'up',
        'k': 'right',
        'p': 'down',
        'tsk': 'left',
        'cluck': 'escape',
    }
    try:
        pyautogui.press(keyboard_mapping[noise_heard])
    except:
        print('No keyboard mapping for "{}"'.format(noise_heard))
    print_noise(noise_heard)


# And define a model that only recognizes these five noises, even if we recorded more earlier:
my_recordings_subset = {k: v for k, v in my_recordings.items() if k in (
    't', 'k', 'p', 'tsk', 'cluck')}
necrodancer_model, _ = build_beatbot(
    device=device, skip_recording=True, starting_noise_data=my_recordings_subset)
